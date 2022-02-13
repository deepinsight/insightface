# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle


@paddle.no_grad()
def sync_params(parameters):
    for param in parameters:
        paddle.distributed.broadcast(
            param.detach(), src=0, group=None, use_calc_stream=True)


@paddle.no_grad()
def sync_gradients(parameters):
    grad_var_set = set()
    grad_vars = []
    sparse_grad_vars = []

    for param in parameters:
        if param.trainable and (param._grad_ivar() is not None):
            g_var = param._grad_ivar()
            assert not g_var._is_sparse(
            ), "Now, it doesn't support sparse parameters"
            grad_vars.append(g_var)
            assert g_var not in grad_var_set
            grad_var_set.add(g_var)

    coalesced_grads_and_vars = \
        paddle.fluid.dygraph.parallel.build_groups(grad_vars, 128 * 1024 * 1024)

    nranks = paddle.distributed.get_world_size()
    for coalesced_grad, _, _ in coalesced_grads_and_vars:
        # need to div nranks
        div_factor = paddle.to_tensor(nranks, dtype=coalesced_grad.dtype)
        paddle.fluid.framework._dygraph_tracer().trace_op(
            type="elementwise_div",
            inputs={'X': coalesced_grad,
                    'Y': div_factor},
            outputs={'Out': coalesced_grad},
            attrs={'axis': -1})

        paddle.distributed.all_reduce(coalesced_grad)

    paddle.fluid.dygraph.parallel._split_tensors(coalesced_grads_and_vars)
