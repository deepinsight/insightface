import logging
import warnings
from collections import namedtuple

import horovod.mxnet as hvd
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np

from default import config
from optimizer import DistributedOptimizer


class SampleDistributeModule(object):
    """
    Large-scale distributed sampling face recognition training Module, of course sampling is an option,
    There will be no loss of accuracy in sampling in large-scale identities training tasks, uses only
    8 NVIDIA RTX2080Ti to complete classification tasks with 10 millions of identities, 64 NVIDIA
    RTX2080Ti can complete classification tasks with 100 million of identities.

    See the original paper:
    https://arxiv.org/abs/2010.05222

    Parameters
    ----------
    symbol: Symbol
        Backbone symbol.
    fc7_model: Object
        Object of margin loss.
    memory_bank: Memory bank Object.
        Object of memory bank, which maintain local class centers and their momentum.
    memory_optimizer: Optimizer object.
        The updater of memory bank, default is sgd optimizer.
    logger:
    """
    def __init__(
        self,
        symbol,
        fc7_model,
        memory_bank,
        memory_optimizer,
        logger=logging,
    ):
        self.size = hvd.size()
        self.rank = hvd.rank()
        self.local_rank = hvd.local_rank()
        self.gpu = mx.gpu(self.local_rank)
        self.cpu = mx.cpu()  # `device_id` is not needed for CPU.
        self.nd_cache = {}
        self.embedding_size = config.embedding_size
        self.batch_size = config.batch_size
        self.num_update = 0
        self.batch_end_param = namedtuple('batch_end_param',
                                          ['loss', 'num_epoch', 'num_update'])

        self.fc7_model = fc7_model
        self.symbol = symbol
        self.logger = logger
        self.backbone_module = mx.module.Module(self.symbol, ['data'],
                                                ['softmax_label'],
                                                logger=self.logger,
                                                context=self.gpu)

        self.memory_bank = memory_bank
        self.memory_optimizer = memory_optimizer
        self.memory_lr = None
        self.loss_cache = None
        self.grad_cache = None

    def forward_backward(self, data_batch):
        """A convenient function that calls both ``forward`` and ``backward``.
        """
        total_feature, total_label = self.forward(data_batch, is_train=True)
        self.backward_all(total_feature, total_label)

    @staticmethod
    def broadcast_parameters(params):
        """
        :param params:
        :return:
        """

        rank_0_dict = {}

        # Run broadcasts.
        for key, tensor in params.items():
            rank_0_dict[key] = hvd.broadcast(tensor, 0, key)
        return rank_0_dict

    def fit(self,
            train_data,
            optimizer_params,
            batch_end_callback,
            initializer,
            arg_params=None,
            aux_params=None):

        # Bind -> Init_params -> Init_optimizers
        self.bind(train_data.provide_data, train_data.provide_label, True)
        self.init_params(initializer, arg_params, aux_params, False)
        self.init_optimizer(optimizer_params=optimizer_params)

        # Sync init
        _arg_params, _aux_params = self.backbone_module.get_params()
        _arg_params_rank_0 = self.broadcast_parameters(_arg_params)
        _aux_params_rank_0 = self.broadcast_parameters(_aux_params)
        self.backbone_module.set_params(_arg_params_rank_0, _aux_params_rank_0)

        # Training loop
        num_epoch = 0
        while True:
            data_iter = iter(train_data)
            end_of_batch = False
            next_data_batch = next(data_iter)

            while not end_of_batch:
                data_batch = next_data_batch
                self.forward_backward(data_batch)
                self.update()
                try:
                    # pre fetch next batch
                    next_data_batch = next(data_iter)
                    self.prepare(next_data_batch, sparse_row_id_fn=None)
                except StopIteration:
                    num_epoch += 1
                    end_of_batch = True
                    logging.info('reset dataset')
                    train_data.reset()

                if batch_end_callback is not None:
                    batch_end_params = self.batch_end_param(
                        loss=self.loss_cache,
                        num_epoch=num_epoch,
                        num_update=self.num_update)
                    batch_end_callback(batch_end_params)

    def get_export_params(self):
        _g, _x = self.backbone_module.get_params()
        g = _g.copy()
        x = _x.copy()
        return g, x

    def get_ndarray2(self, context, name, arr):
        key = "%s_%s" % (name, context)
        if key not in self.nd_cache:
            v = nd.zeros(shape=arr.shape, ctx=context, dtype=arr.dtype)
            self.nd_cache[key] = v
        else:
            v = self.nd_cache[key]
        arr.copyto(v)
        return v

    def get_ndarray(self, context, name, shape, dtype='float32'):
        key = "%s_%s" % (name, context)
        if key not in self.nd_cache:
            v = nd.zeros(shape=shape, ctx=context, dtype=dtype)
            self.nd_cache[key] = v
        else:
            v = self.nd_cache[key]
        return v

    def init_params(self,
                    initializer,
                    arg_params=None,
                    aux_params=None,
                    allow_missing=False,
                    force_init=False,
                    allow_extra=False):
        """Initializes the parameters and auxiliary states.

        Parameters
        ----------
        initializer : Initializer
            Called to initialize parameters if needed.
        arg_params : dict
            If not ``None``, should be a dictionary of existing arg_params. Initialization
            will be copied from that.
        aux_params : dict
            If not ``None``, should be a dictionary of existing aux_params. Initialization
            will be copied from that.
        allow_missing : bool
            If ``True``, params could contain missing values, and the initializer will be
            called to fill those missing params.
        force_init : bool
            If ``True``, will force re-initialize even if already initialized.
        allow_extra : boolean, optional
            Whether allow extra parameters that are not needed by symbol.
            If this is True, no error will be thrown when arg_params or aux_params
            contain extra parameters that is not needed by the executor.
        """
        # backbone
        self.backbone_module.init_params(initializer=initializer,
                                         arg_params=arg_params,
                                         aux_params=aux_params,
                                         allow_missing=allow_missing,
                                         force_init=force_init,
                                         allow_extra=allow_extra)

    def prepare(self, data_batch, sparse_row_id_fn=None):
        if sparse_row_id_fn is not None:
            warnings.warn(
                UserWarning("sparse_row_id_fn is not invoked for BaseModule."))

    def allgather(self, tensor, name, shape, dtype, context):
        """ Implement in-place AllGather using AllReduce
        """
        assert isinstance(tensor, nd.NDArray), type(tensor)
        assert isinstance(name, str), type(name)
        assert isinstance(shape, tuple), type(shape)
        assert isinstance(dtype, str), type(dtype)
        assert isinstance(context, mx.context.Context), type(context)
        total_tensor = self.get_ndarray(context=context,
                                        name=name,
                                        shape=shape,
                                        dtype=dtype)
        total_tensor[:] = 0  # reset array before all-reduce is very important
        total_tensor[self.rank * self.batch_size:self.rank * self.batch_size +
                     self.batch_size] = tensor
        hvd.allreduce_(total_tensor, average=False)  # all-reduce in-place
        return total_tensor

    def forward(self, data_batch, is_train=None):
        self.backbone_module.forward(data_batch, is_train=is_train)
        if is_train:
            self.num_update += 1
            fc1 = self.backbone_module.get_outputs()[0]
            label = data_batch.label[0]

            total_features = self.allgather(tensor=fc1,
                                            name='total_feature',
                                            shape=(self.batch_size * self.size,
                                                   self.embedding_size),
                                            dtype='float32',
                                            context=self.gpu)
            total_labels = self.allgather(tensor=label,
                                          name='total_label',
                                          shape=(self.batch_size *
                                                 self.size, ),
                                          dtype='int32',
                                          context=self.cpu)
            return total_features, total_labels
        else:
            return None

    def backward_all(
        self,
        total_feature,
        total_label,
    ):
        # get memory bank learning rate
        self.memory_lr = self.memory_optimizer.lr_scheduler(self.num_update)

        self.grad_cache = self.get_ndarray(self.gpu, 'grad_cache',
                                           total_feature.shape)
        self.loss_cache = self.get_ndarray(self.gpu, 'loss_cache', [1])

        self.grad_cache[:] = 0
        self.loss_cache[:] = 0

        if not bool(config.sample_ratio - 1):
            grad, loss = self.backward(total_feature, total_label)
        else:
            grad, loss = self.backward_sample(total_feature, total_label)

        self.loss_cache[0] = loss

        total_feature_grad = grad
        total_feature_grad = hvd.allreduce(total_feature_grad, average=False)

        fc1_grad = total_feature_grad[self.batch_size *
                                      self.rank:self.batch_size * self.rank +
                                      self.batch_size]
        self.backbone_module.backward(out_grads=[fc1_grad / self.size])

    def get_outputs(self, merge_multi_context=True):
        """
        Gets outputs of the previous forward computation.

        Returns
        -------
        list of NDArray or list of list of NDArray
            Output.
        """
        return self.backbone_module.get_outputs(
            merge_multi_context=merge_multi_context)

    def update(self):
        """
        Updates parameters according to the installed optimizer and the gradients computed
        in the previous forward-backward batch.
        """
        self.backbone_module.update()
        mx.nd.waitall()

    def bind(self, data_shapes, label_shapes=None, for_training=True):
        self.backbone_module.bind(data_shapes,
                                  label_shapes,
                                  for_training=for_training)

    def init_optimizer(self, optimizer_params, force_init=False):
        """
        Installs and initializes optimizers.

        Parameters
        ----------
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The default value is not a dictionary,
            just to avoid pylint warning of dangerous default values.
        force_init : bool
            Default ``False``, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """
        optimizer_backbone = DistributedOptimizer(
            mx.optimizer.SGD(**optimizer_params))
        self.backbone_module.init_optimizer('local',
                                            optimizer_backbone,
                                            force_init=force_init)

    def backward(self, total_feature, label):
        memory_bank = self.memory_bank
        assert memory_bank.num_local == memory_bank.num_sample, "pass"

        _data = self.get_ndarray2(self.gpu, "data_%d" % self.rank,
                                  total_feature)
        # Attach grad
        _data.attach_grad()
        memory_bank.weight.attach_grad()

        # Convert label
        _label = self.get_ndarray2(self.gpu, 'label_%d' % self.rank, label)
        _label = _label - int(self.rank * memory_bank.num_local)
        _fc7, _one_hot = self.fc7_model.forward(_data,
                                                memory_bank.weight,
                                                mapping_label=_label,
                                                depth=memory_bank.num_local)

        # Sync max
        max_fc7 = nd.max(_fc7, axis=1, keepdims=True)
        max_fc7 = nd.reshape(max_fc7, -1)

        total_max_fc7 = self.get_ndarray(context=self.gpu,
                                         name='total_max_fc7',
                                         shape=(max_fc7.shape[0], self.size),
                                         dtype='float32')
        total_max_fc7[:] = 0
        total_max_fc7[:, self.rank] = max_fc7
        hvd.allreduce_(total_max_fc7, average=False)

        global_max_fc7 = self.get_ndarray(context=self.gpu,
                                          name='global_max_fc7',
                                          shape=(max_fc7.shape[0], 1),
                                          dtype='float32')
        nd.max(total_max_fc7, axis=1, keepdims=True, out=global_max_fc7)

        # Calculate exp(logits)
        _fc7_grad = nd.broadcast_sub(_fc7, global_max_fc7)
        _fc7_grad = nd.exp(_fc7_grad)

        # Calculate sum
        sum_fc7 = nd.sum(_fc7_grad, axis=1, keepdims=True)
        global_sum_fc7 = hvd.allreduce(sum_fc7, average=False)

        # Calculate prob
        _fc7_grad = nd.broadcast_div(_fc7_grad, global_sum_fc7)

        # Calculate loss
        tmp = _fc7_grad * _one_hot
        tmp = nd.sum(tmp, axis=1, keepdims=True)
        tmp = self.get_ndarray2(self.gpu, 'ctx_loss', tmp)
        tmp = hvd.allreduce(tmp, average=False)
        global_loss = -nd.mean(nd.log(tmp + 1e-30))

        # Calculate fc7 grad
        _fc7_grad = _fc7_grad - _one_hot

        # Backward
        _fc7.backward(out_grad=_fc7_grad)

        # Update center
        _weight_grad = memory_bank.weight.grad
        self.memory_optimizer.update(weight=memory_bank.weight,
                                     grad=_weight_grad,
                                     state=memory_bank.weight_mom,
                                     learning_rate=self.memory_lr)

        return _data.grad, global_loss

    def backward_sample(self, total_feature, label):
        this_rank_classes = int(self.memory_bank.num_sample)
        local_index, unique_sorted_global_label = self.memory_bank.sample(
            label)

        # Get local index
        _mapping_dict = {}
        local_sampled_class = local_index + self.rank * self.memory_bank.num_local
        global_label_set = set(unique_sorted_global_label)
        for idx, absolute_label in enumerate(local_sampled_class):
            if absolute_label in global_label_set:
                _mapping_dict[
                    absolute_label] = idx + self.rank * self.memory_bank.num_sample

        label_list = list(label.asnumpy())
        mapping_label = []
        for i in range(len(label_list)):
            absolute_label = label_list[i]
            if absolute_label in _mapping_dict.keys():
                mapping_label.append(_mapping_dict[absolute_label])
            else:
                mapping_label.append(-1)

        mapping_label = nd.array(mapping_label, dtype=np.int32)

        # Get weight
        local_index = nd.array(local_index)
        local_index = self.get_ndarray2(self.gpu, "local_index", local_index)
        sample_weight, sample_weight_mom = self.memory_bank.get(local_index)

        # Sync to gpu
        if self.memory_bank.gpu:
            _data = self.get_ndarray2(self.gpu, "data_%d" % self.rank,
                                      total_feature)
            _weight = self.get_ndarray2(self.gpu, 'weight_%d' % self.rank,
                                        sample_weight)
            _weight_mom = self.get_ndarray2(self.gpu,
                                            'weight_mom_%d' % self.rank,
                                            sample_weight_mom)
        else:
            _data = self.get_ndarray2(self.gpu, "data_%d" % self.rank,
                                      total_feature)
            _weight = self.get_ndarray2(self.gpu, 'weight_%d' % self.rank,
                                        sample_weight)
            _weight_mom = self.get_ndarray2(self.gpu,
                                            'weight_mom_%d' % self.rank,
                                            sample_weight_mom)

        # Attach grad
        _data.attach_grad()
        _weight.attach_grad()

        # Convert label
        _label = self.get_ndarray2(self.gpu, 'mapping_label_%d' % self.rank,
                                   mapping_label)
        _label = _label - int(self.rank * self.memory_bank.num_sample)
        _fc7, _one_hot = self.fc7_model.forward(_data,
                                                _weight,
                                                mapping_label=_label,
                                                depth=this_rank_classes)

        # Sync max
        max_fc7 = nd.max(_fc7, axis=1, keepdims=True)
        max_fc7 = nd.reshape(max_fc7, -1)

        total_max_fc7 = self.get_ndarray(context=self.gpu,
                                         name='total_max_fc7',
                                         shape=(max_fc7.shape[0], self.size),
                                         dtype='float32')
        total_max_fc7[:] = 0
        total_max_fc7[:, self.rank] = max_fc7
        hvd.allreduce_(total_max_fc7, average=False)

        global_max_fc7 = self.get_ndarray(context=self.gpu,
                                          name='global_max_fc7',
                                          shape=(max_fc7.shape[0], 1),
                                          dtype='float32')
        nd.max(total_max_fc7, axis=1, keepdims=True, out=global_max_fc7)

        # Calculate exp(logits)
        _fc7_grad = nd.broadcast_sub(_fc7, global_max_fc7)
        _fc7_grad = nd.exp(_fc7_grad)

        # Calculate sum
        sum_fc7 = nd.sum(_fc7_grad, axis=1, keepdims=True)
        global_sum_fc7 = hvd.allreduce(sum_fc7, average=False)

        # Calculate grad
        _fc7_grad = nd.broadcast_div(_fc7_grad, global_sum_fc7)

        # Calculate loss
        tmp = _fc7_grad * _one_hot
        tmp = nd.sum(tmp, axis=1, keepdims=True)
        tmp = self.get_ndarray2(self.gpu, 'ctx_loss', tmp)
        tmp = hvd.allreduce(tmp, average=False)
        global_loss = -nd.mean(nd.log(tmp + 1e-30))

        _fc7_grad = _fc7_grad - _one_hot

        # Backward
        _fc7.backward(out_grad=_fc7_grad)

        # Update center
        _weight_grad = _weight.grad
        self.memory_optimizer.update(weight=_weight,
                                     grad=_weight_grad,
                                     state=_weight_mom,
                                     learning_rate=self.memory_lr)
        if self.memory_bank.gpu:
            self.memory_bank.set(index=local_index,
                                 updated_weight=_weight,
                                 updated_weight_mom=_weight_mom)
        else:
            self.memory_bank.set(index=local_index,
                                 updated_weight=self.get_ndarray2(
                                     mx.cpu(), "cpu_weight_%d" % self.rank,
                                     _weight),
                                 updated_weight_mom=self.get_ndarray2(
                                     mx.cpu(), "cpu_weight_mom_%d" % self.rank,
                                     _weight_mom))
        return _data.grad, global_loss
