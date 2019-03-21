
'''
@author: insightface
'''

import logging
import copy
import time
import os

import mxnet as mx
import numpy as np
from mxnet import context as ctx
from mxnet.initializer import Uniform
from mxnet.module.base_module import BaseModule
from mxnet.module.module import Module
from mxnet import metric
from mxnet.model import BatchEndParam
from mxnet import io
import mxnet.ndarray as nd
from config import config

class ParallModule(BaseModule):
    def __init__(self, symbol, data_names, label_names,
                 logger=logging, context=ctx.cpu(), work_load_list=None,
                 asymbol = None,
                 args = None):
        super(ParallModule, self).__init__(logger=logger)
        self._symbol = symbol
        self._asymbol = asymbol
        self._data_names = data_names
        self._label_names = label_names
        self._context = context
        self._work_load_list = work_load_list
        self._num_classes = config.num_classes
        self._batch_size = args.batch_size
        self._verbose = args.verbose
        self._emb_size = config.emb_size
        self._local_class_start = args.local_class_start
        self._iter = 0

        self._curr_module = None

        self._num_workers = config.num_workers
        self._num_ctx = len(self._context)
        self._ctx_num_classes = args.ctx_num_classes
        self._nd_cache = {}
        self._ctx_cpu = mx.cpu()
        self._ctx_single_gpu = self._context[-1]
        self._fixed_param_names = None
        self._curr_module = Module(self._symbol, self._data_names, self._label_names, logger=self.logger,
                        context=self._context, work_load_list=self._work_load_list,
                        fixed_param_names=self._fixed_param_names)
        self._arcface_modules = []
        self._ctx_class_start = []
        for i in range(len(self._context)):

          args._ctxid = i
          _module = Module(self._asymbol(args), self._data_names, self._label_names, logger=self.logger,
                          context=mx.gpu(i), work_load_list=self._work_load_list,
                          fixed_param_names=self._fixed_param_names)
          self._arcface_modules.append(_module)
          _c = args.local_class_start + i*args.ctx_num_classes
          self._ctx_class_start.append(_c)
        self._usekv = False
        if self._usekv:
          self._distkv = mx.kvstore.create('dist_sync')
          self._kvinit = {}


    def _reset_bind(self):
        self.binded = False
        self._curr_module = None

    @property
    def data_names(self):
        return self._data_names

    @property
    def output_names(self):
        return self._symbol.list_outputs()

    @property
    def data_shapes(self):
        assert self.binded
        return self._curr_module.data_shapes

    @property
    def label_shapes(self):
        assert self.binded
        return self._curr_module.label_shapes

    @property
    def output_shapes(self):
        assert self.binded
        return self._curr_module.output_shapes

    def get_export_params(self):
        assert self.binded and self.params_initialized
        _g, _x = self._curr_module.get_params()
        g = _g.copy()
        x = _x.copy()
        return g, x

    def get_params(self):
        assert self.binded and self.params_initialized
        _g, _x = self._curr_module.get_params()
        g = _g.copy()
        x = _x.copy()
        for _module in self._arcface_modules:
          _g, _x = _module.get_params()
          ag = _g.copy()
          ax = _x.copy()
          g.update(ag)
          x.update(ax)
        return g, x

    def set_params(self, arg_params, aux_params, allow_missing=False, force_init=True,
                   allow_extra=False):
      g = arg_params
      x = aux_params
      #ag = {}
      #ax = {}
      rk = []
      for k,v in g.iteritems():
        if k.startswith('fc7'):
          p1 = k.find('_')
          p2 = k.rfind('_')
          _ctxid = int(k[p1+1:p2])
          self._arcface_modules[_ctxid].set_params({k:v}, {})
          rk.append(k)
      for k in rk:
        del g[k]
      #for k,v in g.iteritems():
      #  print('g', k, v.shape)
      #for k,v in ag.iteritems():
      #  print('ag', k, v.shape)
      self._curr_module.set_params(g, x)
      #self._arcface_module.set_params(ag, ax)


    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False, allow_extra=False):
        if self.params_initialized and not force_init:
            return
        assert self.binded, 'call bind before initializing the parameters'
        #TODO init the same weights with all work nodes
        self._curr_module.init_params(initializer=initializer, arg_params=None,
                                      aux_params=None, allow_missing=allow_missing,
                                      force_init=force_init, allow_extra=allow_extra)
        for _module in self._arcface_modules:
          #_initializer = initializer
          _initializer = mx.init.Normal(0.01)
          _module.init_params(initializer=_initializer, arg_params=None,
                                        aux_params=None, allow_missing=allow_missing,
                                        force_init=force_init, allow_extra=allow_extra)
        self.params_initialized = True


    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None):
        print('in_bind', self.params_initialized, data_shapes, label_shapes)
        if self.params_initialized:
            arg_params, aux_params = self.get_params()

        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return

        assert shared_module is None, 'shared_module for MutableModule is not supported'
        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True
        self._curr_module.bind(data_shapes, label_shapes, for_training, inputs_need_grad,
                    force_rebind=False, shared_module=None)
        _data_shape = data_shapes[0][1]
        print('_data_shape', _data_shape, label_shapes)
        for _module in self._arcface_modules:
          _module.bind([('data', (_data_shape[0]*self._num_workers, self._emb_size))], [('softmax_label', (_data_shape[0]*self._num_workers,))], for_training, True,
                      force_rebind=False, shared_module=None)
        if self.params_initialized:
            self.set_params(arg_params, aux_params)

    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        assert self.binded and self.params_initialized
        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring.')
            return

        self._curr_module.init_optimizer(kvstore, optimizer, optimizer_params,
                                         force_init=force_init)
        for _module in self._arcface_modules:
          _module.init_optimizer(kvstore, optimizer, optimizer_params,
                                           force_init=force_init)
        self.optimizer_initialized = True

    def kv_push(self, key, value):
      #if value.context!=mx.cpu():
      #  value = value.as_in_context(mx.cpu())
      if not key in self._kvinit:
        self._distkv.init(key, nd.zeros_like(value))
        self._kvinit[key] = 1
      self._distkv.push(key, value)

    #get fc1 and partial fc7
    def forward(self, data_batch, is_train=None):
        #g,x = self.get_params()
        #print('{fc7_weight[0][0]}', self._iter, g['fc7_0_weight'].asnumpy()[0][0])
        #print('{pre_fc1_weight[0][0]}', self._iter, g['pre_fc1_weight'].asnumpy()[0][0])


        assert self.binded and self.params_initialized
        self._curr_module.forward(data_batch, is_train=is_train)
        if is_train:
          self._iter+=1
          fc1, label = self._curr_module.get_outputs(merge_multi_context=True)
          global_fc1 = fc1
          self.global_label = label.as_in_context(self._ctx_cpu)


          for i, _module in enumerate(self._arcface_modules):
            _label = self.global_label - self._ctx_class_start[i]
            db_global_fc1 = io.DataBatch([global_fc1], [_label])
            _module.forward(db_global_fc1) #fc7 with margin
        #print('forward end')


    def get_ndarray(self, context, name, shape):
      key = "%s_%s"%(name, context)
      #print(key)
      if not key in self._nd_cache:
        v = nd.zeros( shape=shape, ctx = context)
        self._nd_cache[key] = v
      else:
        v = self._nd_cache[key]
      return v

    def get_ndarray2(self, context, name, arr):
      key = "%s_%s"%(name, context)
      #print(key)
      if not key in self._nd_cache:
        v = nd.zeros( shape=arr.shape, ctx = context)
        self._nd_cache[key] = v
      else:
        v = self._nd_cache[key]
      arr.copyto(v)
      return v

    def backward(self, out_grads=None):
        #print('in backward')
        assert self.binded and self.params_initialized
        #tmp_ctx = self._ctx_cpu
        tmp_ctx = self._ctx_single_gpu
        fc7_outs = []
        ctx_fc7_max = self.get_ndarray(tmp_ctx, 'ctx_fc7_max', (self._batch_size, len(self._context)))
        #local_fc7_max = nd.zeros( (self.global_label.shape[0],1), ctx=mx.cpu())
        for i, _module in enumerate(self._arcface_modules):
          _fc7 = _module.get_outputs(merge_multi_context=True)[0]
          fc7_outs.append(_fc7)
          _fc7_max = nd.max(_fc7, axis=1).as_in_context(tmp_ctx)
          ctx_fc7_max[:,i] = _fc7_max

        local_fc7_max = self.get_ndarray(tmp_ctx, 'local_fc7_max', (self._batch_size, 1))
        nd.max(ctx_fc7_max, axis=1, keepdims=True, out=local_fc7_max)
        global_fc7_max = local_fc7_max
        #local_fc7_sum = None
        local_fc7_sum = self.get_ndarray(tmp_ctx, 'local_fc7_sum', (self._batch_size,1))
        local_fc7_sum[:,:] = 0.0
        for i, _module in enumerate(self._arcface_modules):
          _max = self.get_ndarray2(fc7_outs[i].context, 'fc7_max', global_fc7_max)
          fc7_outs[i] = nd.broadcast_sub(fc7_outs[i], _max)
          fc7_outs[i] = nd.exp(fc7_outs[i])
          _sum = nd.sum(fc7_outs[i], axis=1, keepdims=True).as_in_context(tmp_ctx)
          local_fc7_sum += _sum
        global_fc7_sum = local_fc7_sum

        if self._iter%self._verbose==0:
          #_ctx = self._context[-1]
          _ctx = self._ctx_cpu
          _probs = []
          for i, _module in enumerate(self._arcface_modules):
            _prob = self.get_ndarray2(_ctx, '_fc7_prob_%d'%i, fc7_outs[i])
            _probs.append(_prob)
          fc7_prob = self.get_ndarray(_ctx, 'test_fc7_prob', (self._batch_size, self._ctx_num_classes*len(self._context)))
          nd.concat(*_probs, dim=1, out=fc7_prob)
          fc7_pred = nd.argmax(fc7_prob, axis=1)
          local_label = self.global_label - self._local_class_start
          #local_label = self.get_ndarray2(_ctx, 'test_label', local_label)
          _pred = nd.equal(fc7_pred, local_label)
          print('{fc7_acc}', self._iter, nd.mean(_pred).asnumpy()[0])


        #local_fc1_grad = []
        #fc1_grad_ctx = self._ctx_cpu
        fc1_grad_ctx = self._ctx_single_gpu
        local_fc1_grad = self.get_ndarray(fc1_grad_ctx, 'local_fc1_grad', (self._batch_size,self._emb_size))
        local_fc1_grad[:,:] = 0.0

        for i, _module in enumerate(self._arcface_modules):
          _sum = self.get_ndarray2(fc7_outs[i].context, 'fc7_sum', global_fc7_sum)
          fc7_outs[i] = nd.broadcast_div(fc7_outs[i], _sum)
          a = i*self._ctx_num_classes
          b = (i+1)*self._ctx_num_classes
          _label = self.global_label - self._ctx_class_start[i]
          _label = self.get_ndarray2(fc7_outs[i].context, 'label', _label)
          onehot_label = self.get_ndarray(fc7_outs[i].context, 'label_onehot', (self._batch_size, self._ctx_num_classes))
          nd.one_hot(_label, depth=self._ctx_num_classes, on_value = 1.0, off_value = 0.0, out=onehot_label)
          fc7_outs[i] -= onehot_label
          _module.backward(out_grads = [fc7_outs[i]])
          #ctx_fc1_grad = _module.get_input_grads()[0].as_in_context(mx.cpu())
          ctx_fc1_grad = self.get_ndarray2(fc1_grad_ctx, 'ctx_fc1_grad_%d'%i, _module.get_input_grads()[0])
          local_fc1_grad += ctx_fc1_grad

        global_fc1_grad = local_fc1_grad
        self._curr_module.backward(out_grads = [global_fc1_grad])


    def update(self):
        assert self.binded and self.params_initialized and self.optimizer_initialized
        self._curr_module.update()
        for i, _module in enumerate(self._arcface_modules):
          _module.update()
        mx.nd.waitall()


    def get_outputs(self, merge_multi_context=True):
        assert self.binded and self.params_initialized
        return self._curr_module.get_outputs(merge_multi_context=merge_multi_context)
        #return self._arcface_module.get_outputs(merge_multi_context=merge_multi_context)

    def get_input_grads(self, merge_multi_context=True):
        assert self.binded and self.params_initialized and self.inputs_need_grad
        return self._curr_module.get_input_grads(merge_multi_context=merge_multi_context)

    def update_metric(self, eval_metric, labels):
        assert self.binded and self.params_initialized
        #self._curr_module.update_metric(eval_metric, labels)
        #label = labels[0]
        #print(label.shape)
        #self._arcface_module.update_metric(eval_metric, labels)

    def install_monitor(self, mon):
        """ Install monitor on all executors """
        assert self.binded
        self._curr_module.install_monitor(mon)

    def forward_backward(self, data_batch):
        """A convenient function that calls both ``forward`` and ``backward``."""
        self.forward(data_batch, is_train=True) # get fc1 and partial fc7
        self.backward()

    def fit(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None, sparse_row_id_fn=None):
        """Trains the module parameters.

        Checkout `Module Tutorial <http://mxnet.io/tutorials/basic/module.html>`_ to see
        a end-to-end use-case.

        Parameters
        ----------
        train_data : DataIter
            Train DataIter.
        eval_data : DataIter
            If not ``None``, will be used as validation set and the performance
            after each epoch will be evaluated.
        eval_metric : str or EvalMetric
            Defaults to 'accuracy'. The performance measure used to display during training.
            Other possible predefined metrics are:
            'ce' (CrossEntropy), 'f1', 'mae', 'mse', 'rmse', 'top_k_accuracy'.
        epoch_end_callback : function or list of functions
            Each callback will be called with the current `epoch`, `symbol`, `arg_params`
            and `aux_params`.
        batch_end_callback : function or list of function
            Each callback will be called with a `BatchEndParam`.
        kvstore : str or KVStore
            Defaults to 'local'.
        optimizer : str or Optimizer
            Defaults to 'sgd'.
        optimizer_params : dict
            Defaults to ``(('learning_rate', 0.01),)``. The parameters for
            the optimizer constructor.
            The default value is not a dict, just to avoid pylint warning on dangerous
            default values.
        eval_end_callback : function or list of function
            These will be called at the end of each full evaluation, with the metrics over
            the entire evaluation set.
        eval_batch_end_callback : function or list of function
            These will be called at the end of each mini-batch during evaluation.
        initializer : Initializer
            The initializer is called to initialize the module parameters when they are
            not already initialized.
        arg_params : dict
            Defaults to ``None``, if not ``None``, should be existing parameters from a trained
            model or loaded from a checkpoint (previously saved model). In this case,
            the value here will be used to initialize the module parameters, unless they
            are already initialized by the user via a call to `init_params` or `fit`.
            `arg_params` has a higher priority than `initializer`.
        aux_params : dict
            Defaults to ``None``. Similar to `arg_params`, except for auxiliary states.
        allow_missing : bool
            Defaults to ``False``. Indicates whether to allow missing parameters when `arg_params`
            and `aux_params` are not ``None``. If this is ``True``, then the missing parameters
            will be initialized via the `initializer`.
        force_rebind : bool
            Defaults to ``False``. Whether to force rebinding the executors if already bound.
        force_init : bool
            Defaults to ``False``. Indicates whether to force initialization even if the
            parameters are already initialized.
        begin_epoch : int
            Defaults to 0. Indicates the starting epoch. Usually, if resumed from a
            checkpoint saved at a previous training phase at epoch N, then this value should be
            N+1.
        num_epoch : int
            Number of epochs for training.
        sparse_row_id_fn : A callback function
            The function  takes `data_batch` as an input and returns a dict of
            str -> NDArray. The resulting dict is used for pulling row_sparse
            parameters from the kvstore, where the str key is the name of the param,
            and the value is the row id of the param to pull.

        Examples
        --------
        >>> # An example of using fit for training.
        >>> # Assume training dataIter and validation dataIter are ready
        >>> # Assume loading a previously checkpointed model
        >>> sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 3)
        >>> mod.fit(train_data=train_dataiter, eval_data=val_dataiter, optimizer='sgd',
        ...     optimizer_params={'learning_rate':0.01, 'momentum': 0.9},
        ...     arg_params=arg_params, aux_params=aux_params,
        ...     eval_metric='acc', num_epoch=10, begin_epoch=3)
        """
        assert num_epoch is not None, 'please specify number of epochs'
        assert arg_params is None and aux_params is None

        self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
                  for_training=True, force_rebind=force_rebind)
        if monitor is not None:
            self.install_monitor(monitor)
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)
        epoch_eval_metric = copy.deepcopy(eval_metric)

        ################################################################################
        # training loop
        ################################################################################
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            eval_metric.reset()
            epoch_eval_metric.reset()
            nbatch = 0
            data_iter = iter(train_data)
            end_of_batch = False
            next_data_batch = next(data_iter)
            while not end_of_batch:
                data_batch = next_data_batch
                if monitor is not None:
                    monitor.tic()
                self.forward_backward(data_batch)
                self.update()
                assert not isinstance(data_batch, list)

                #if isinstance(data_batch, list):
                #    #print('XXX')
                #    self.update_metric(eval_metric,
                #                       [db.label for db in data_batch],
                #                       pre_sliced=True)
                #    self.update_metric(epoch_eval_metric,
                #                       [db.label for db in data_batch],
                #                       pre_sliced=True)
                #else:
                #    #print('before update metric')
                #    self.update_metric(eval_metric, data_batch.label)
                #    self.update_metric(epoch_eval_metric, data_batch.label)
                #labels = data_batch.label
                #labels = [self.global_label]
                #self.update_metric(eval_metric, labels)
                #self.update_metric(epoch_eval_metric, labels)

                try:
                    # pre fetch next batch
                    next_data_batch = next(data_iter)
                    self.prepare(next_data_batch, sparse_row_id_fn=sparse_row_id_fn)
                except StopIteration:
                    end_of_batch = True

                if monitor is not None:
                    monitor.toc_print()

                #if end_of_batch:
                #    eval_name_vals = epoch_eval_metric.get_name_value()

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=None,
                                                     locals=locals())
                    batch_end_callback(batch_end_params)
                    #for callback in _as_list(batch_end_callback):
                    #    callback(batch_end_params)
                nbatch += 1

            # one epoch of training is finished
            #for name, val in eval_name_vals:
            #    self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            # sync aux params across devices
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)

            # end of 1 epoch, reset the data-iter for another epoch
            train_data.reset()

