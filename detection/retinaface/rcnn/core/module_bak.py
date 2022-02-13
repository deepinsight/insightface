"""A `MutableModule` implement the `BaseModule` API, and allows input shape
varying with training iterations. If shapes vary, executors will rebind,
using shared arrays from the initial module binded with maximum shape.
"""

import logging

from mxnet import context as ctx
from mxnet.initializer import Uniform
from mxnet.module.base_module import BaseModule
from mxnet.module.module import Module


class MutableModule(BaseModule):
    """A mutable module is a module that supports variable input data.

    Parameters
    ----------
    symbol : Symbol
    data_names : list of str
    label_names : list of str
    logger : Logger
    context : Context or list of Context
    work_load_list : list of number
    max_data_shapes : list of (name, shape) tuple, designating inputs whose shape vary
    max_label_shapes : list of (name, shape) tuple, designating inputs whose shape vary
    fixed_param_prefix : list of str, indicating fixed parameters
    """
    def __init__(self,
                 symbol,
                 data_names,
                 label_names,
                 logger=logging,
                 context=ctx.cpu(),
                 work_load_list=None,
                 max_data_shapes=None,
                 max_label_shapes=None,
                 fixed_param_prefix=None):
        super(MutableModule, self).__init__(logger=logger)
        self._symbol = symbol
        self._data_names = data_names
        self._label_names = label_names
        self._context = context
        self._work_load_list = work_load_list

        self._curr_module = None
        self._max_data_shapes = max_data_shapes
        self._max_label_shapes = max_label_shapes
        self._fixed_param_prefix = fixed_param_prefix

        fixed_param_names = list()
        if fixed_param_prefix is not None:
            for name in self._symbol.list_arguments():
                for prefix in self._fixed_param_prefix:
                    if prefix in name:
                        fixed_param_names.append(name)
        self._fixed_param_names = fixed_param_names

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

    def get_params(self):
        assert self.binded and self.params_initialized
        return self._curr_module.get_params()

    def init_params(self,
                    initializer=Uniform(0.01),
                    arg_params=None,
                    aux_params=None,
                    allow_missing=False,
                    force_init=False,
                    allow_extra=False):
        if self.params_initialized and not force_init:
            return
        assert self.binded, 'call bind before initializing the parameters'
        self._curr_module.init_params(initializer=initializer,
                                      arg_params=arg_params,
                                      aux_params=aux_params,
                                      allow_missing=allow_missing,
                                      force_init=force_init,
                                      allow_extra=allow_extra)
        self.params_initialized = True

    def bind(self,
             data_shapes,
             label_shapes=None,
             for_training=True,
             inputs_need_grad=False,
             force_rebind=False,
             shared_module=None,
             grad_req='write'):
        # in case we already initialized params, keep it
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

        max_shapes_dict = dict()
        if self._max_data_shapes is not None:
            max_shapes_dict.update(dict(self._max_data_shapes))
        if self._max_label_shapes is not None:
            max_shapes_dict.update(dict(self._max_label_shapes))

        max_data_shapes = list()
        for name, shape in data_shapes:
            if name in max_shapes_dict:
                max_data_shapes.append((name, max_shapes_dict[name]))
            else:
                max_data_shapes.append((name, shape))

        max_label_shapes = list()
        if label_shapes is not None:
            for name, shape in label_shapes:
                if name in max_shapes_dict:
                    max_label_shapes.append((name, max_shapes_dict[name]))
                else:
                    max_label_shapes.append((name, shape))

        if len(max_label_shapes) == 0:
            max_label_shapes = None

        module = Module(self._symbol,
                        self._data_names,
                        self._label_names,
                        logger=self.logger,
                        context=self._context,
                        work_load_list=self._work_load_list,
                        fixed_param_names=self._fixed_param_names)
        module.bind(max_data_shapes,
                    max_label_shapes,
                    for_training,
                    inputs_need_grad,
                    force_rebind=False,
                    shared_module=None)
        self._curr_module = module

        # copy back saved params, if already initialized
        if self.params_initialized:
            self.set_params(arg_params, aux_params)

    def init_optimizer(self,
                       kvstore='local',
                       optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01), ),
                       force_init=False):
        assert self.binded and self.params_initialized
        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring.')
            return

        self._curr_module.init_optimizer(kvstore,
                                         optimizer,
                                         optimizer_params,
                                         force_init=force_init)
        self.optimizer_initialized = True

    def forward(self, data_batch, is_train=None):
        assert self.binded and self.params_initialized

        # get current_shapes
        if self._curr_module.label_shapes is not None:
            current_shapes = dict(self._curr_module.data_shapes +
                                  self._curr_module.label_shapes)
        else:
            current_shapes = dict(self._curr_module.data_shapes)

        # get input_shapes
        if data_batch.provide_label is not None:
            input_shapes = dict(data_batch.provide_data +
                                data_batch.provide_label)
        else:
            input_shapes = dict(data_batch.provide_data)

        # decide if shape changed
        shape_changed = False
        for k, v in current_shapes.items():
            if v != input_shapes[k]:
                shape_changed = True

        if shape_changed:
            module = Module(self._symbol,
                            self._data_names,
                            self._label_names,
                            logger=self.logger,
                            context=self._context,
                            work_load_list=self._work_load_list,
                            fixed_param_names=self._fixed_param_names)
            module.bind(data_batch.provide_data,
                        data_batch.provide_label,
                        self._curr_module.for_training,
                        self._curr_module.inputs_need_grad,
                        force_rebind=False,
                        shared_module=self._curr_module)
            self._curr_module = module

        self._curr_module.forward(data_batch, is_train=is_train)

    def backward(self, out_grads=None):
        assert self.binded and self.params_initialized
        self._curr_module.backward(out_grads=out_grads)

    def update(self):
        assert self.binded and self.params_initialized and self.optimizer_initialized
        self._curr_module.update()

    def get_outputs(self, merge_multi_context=True):
        assert self.binded and self.params_initialized
        return self._curr_module.get_outputs(
            merge_multi_context=merge_multi_context)

    def get_input_grads(self, merge_multi_context=True):
        assert self.binded and self.params_initialized and self.inputs_need_grad
        return self._curr_module.get_input_grads(
            merge_multi_context=merge_multi_context)

    def update_metric(self, eval_metric, labels):
        assert self.binded and self.params_initialized
        self._curr_module.update_metric(eval_metric, labels)

    def install_monitor(self, mon):
        """ Install monitor on all executors """
        assert self.binded
        self._curr_module.install_monitor(mon)
