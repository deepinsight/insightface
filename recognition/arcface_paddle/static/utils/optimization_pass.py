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


def check_contains(name, name_list):
    for n in name_list:
        if name in n:
            return True
    return False


def gather_optimization_pass(program, weight_name):
    op_idxs = []
    gather_grad_op = None
    momentum_op = None
    for idx, op in enumerate(program.global_block().ops):
        if (op.type == 'gather_grad' or
                op.type == 'momentum') and check_contains(weight_name,
                                                          op.input_arg_names):
            op_idxs.append(idx)
            if op.type == 'momentum':
                momentum_op = op
            if op.type == 'gather_grad':
                gather_grad_op = op

    if gather_grad_op is not None and momentum_op is not None:
        inputs = {
            'Param': momentum_op.input('Param'),
            'Velocity': momentum_op.input('Velocity'),
            'LearningRate': momentum_op.input('LearningRate'),
            'Grad': gather_grad_op.input('Out@GRAD'),
            'Index': gather_grad_op.input('Index'),
            'Axis': gather_grad_op.input('Axis'),
        }
        outputs = {
            'ParamOut': momentum_op.output('ParamOut'),
            'VelocityOut': momentum_op.output('VelocityOut'),
        }
        if 'MasterParam' in momentum_op.input_names and len(
                momentum_op.input('MasterParam')) > 0:
            inputs['MasterParam'] = momentum_op.input('MasterParam')
        if 'MasterParamOut' in momentum_op.output_names and len(
                momentum_op.output('MasterParamOut')) > 0:
            outputs['MasterParamOut'] = momentum_op.output('MasterParamOut')

        attrs = {
            'mu': momentum_op.attr('mu'),
            'use_nesterov': momentum_op.attr('use_nesterov'),
            'regularization_method': momentum_op.attr('regularization_method'),
            'regularization_coeff': momentum_op.attr('regularization_coeff'),
            'multi_precision': momentum_op.attr('multi_precision'),
            'rescale_grad': momentum_op.attr('rescale_grad'),
            'op_device': momentum_op.attr('op_device'),
            'op_namescope': momentum_op.attr('op_namescope'),
            'op_role': momentum_op.attr('op_role'),
            'op_role_var': momentum_op.input('Param'),
            'axis': gather_grad_op.attr('axis'),
        }
        program.global_block()._insert_op(
            op_idxs[-1] + 1,
            type='sparse_momentum',
            inputs=inputs,
            outputs=outputs,
            attrs=attrs)

        for idx in reversed(op_idxs):
            program.global_block()._remove_op(idx, sync=False)

        var_names = []
        for idx, name in enumerate(program.global_block().vars):
            if '@GRAD' in name and weight_name in name:
                var_names.append(name)
        for name in var_names:
            program.global_block()._remove_var(name, sync=False)
        program.global_block()._sync_with_cpp()


def amp_pass(program, weight_name):
    for idx, op in enumerate(program.global_block().ops):
        if (op.type == 'update_loss_scaling' or
                op.type == 'check_finite_and_unscale'):
            input_idxs = []
            input_arg_names = op.input("X")
            # input_arg_names.append(gather_grad_op.input('Out@GRAD')[0])
            for i, name in enumerate(input_arg_names):
                if '@GRAD' in name and weight_name in name:
                    input_idxs.append(i)
            if len(input_idxs) > 0:
                for i in reversed(input_idxs):
                    input_arg_names.pop(i)
                op.desc.set_input("X", input_arg_names)

            output_idxs = []
            output_arg_names = op.output("Out")
            # output_arg_names.append(gather_grad_op.input('Out@GRAD')[0])
            for i, name in enumerate(output_arg_names):
                if '@GRAD' in name and weight_name in name:
                    output_idxs.append(i)
            if len(output_idxs) > 0:
                for i in reversed(output_idxs):
                    output_arg_names.pop(i)
                op.desc.set_output("Out", output_arg_names)

            if op.type == 'check_finite_and_unscale':
                op_role_idxs = []
                op_role_var = op.attr("op_role_var")
                for i, name in enumerate(op_role_var):
                    if '@GRAD' in name and weight_name in name:
                        op_role_idxs.append(i)
                if len(op_role_idxs) > 0:
                    for i in reversed(op_role_idxs):
                        op_role_var.pop(i)
                    op.desc._set_attr("op_role_var", op_role_var)
