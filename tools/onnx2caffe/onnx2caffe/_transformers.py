from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Sequence, Text, Dict, List
import numpy as np

from onnx import TensorProto

from ._graph import Graph, Node


class NodesFuser(object):
    '''
    An abstract helper for merging nodes
    '''
    def __init__(self,
                 num_nodes,  # type: int
                 ):
        # type: (...) -> None
        assert num_nodes >= 2, "Algorithm only works if fusing multiple nodes"
        self.num_nodes = num_nodes

    def __call__(self, graph):  # type: (Graph) -> Graph
        nodes = graph.nodes
        merged_nodes = {}
        for node in nodes:
            nodes_window = []  # type: List[Node]
            n = node
            for _ in range(self.num_nodes - 1):
                if len(n.parents) != 1:
                    # We're only fusing nodes with single parents
                    break
                p = n.get_only_parent()
                if len(p.children) != 1:
                    # We can only fuse a node if its parent's
                    # value isn't used by any other node.
                    break
                nodes_window.insert(0, n)
                n = p
            if len(nodes_window) > 0:
                # add parent of chained nodes
                first = nodes_window[0]
                p = first.get_only_parent()
                if len(p.children) == 1:
                    nodes_window.insert(0, p)
            if len(nodes_window) != self.num_nodes:
                continue
            if not self.is_eligible(graph, nodes_window):
                continue
            merged = self.merge(graph, nodes_window)
            first, last = nodes_window[0], nodes_window[-1]
            for parent in first.parents:
                parent.children.remove(first)
                if merged[0] not in parent.children:
                    parent.add_child(merged[0])
            for child in last.children:
                child.parents.remove(last)
                if merged[-1] not in child.parents:
                    child.add_parent(merged[-1])
            for n in nodes_window:
                merged_nodes[n.name] = merged

        transformed_nodes = []
        added_merged = []  # type: List[Node]
        for node in nodes:
            if node.name in merged_nodes:
                merged = merged_nodes[node.name]
                if merged[0] not in added_merged:
                    for n in merged:
                        transformed_nodes.append(n)
                    added_merged.append(merged[0])
            else:
                transformed_nodes.append(node)
        return Graph(transformed_nodes, graph.inputs, graph.outputs, graph.shape_dict)

    def is_eligible(self, graph, nodes):  # type: (Graph, Sequence[Node]) -> bool
        '''Returns true if this subset of nodes is eligible for fusion.'''
        raise NotImplementedError('Must be implemented by subclass.')

    def merge(self, graph, nodes):  # type: (Graph, Sequence[Node]) -> Sequence[Node]
        '''Merge nodes'''
        nodes[0].outputs = nodes[-1].outputs
        return [nodes[0]]


class ConvAddFuser(NodesFuser):
    '''
    Fuses Add layer into parent convolution layer.
    '''
    def __init__(self):  # type: () -> None
        super(ConvAddFuser, self).__init__(2)

    def is_eligible(self, graph, nodes):  # type: (Graph, Sequence[Node]) -> bool
        parent, child = nodes[0], nodes[1]
        if parent.op_type != 'Conv':
            return False
        if child.op_type != 'Add':
            return False
        if 'broadcast' not in child.attrs:
            return False
        if 'axis' not in child.attrs:
            return False
        if parent.inputs[1] not in parent.input_tensors:
            return False
        if len(parent.inputs) > 2 and parent.inputs[2] not in parent.input_tensors:
            return False
        if child.inputs[1] not in child.input_tensors:
            return False

        broadcast = child.attrs['broadcast']
        if broadcast != 1:
            return False

        axis = child.attrs['axis']
        if axis != 1:
            return False

        return True

    def merge(self, graph, nodes):  # type: (Graph, Sequence[Node]) -> Sequence[Node]
        parent, child = nodes[0], nodes[1]
        output_channels = parent.input_tensors[parent.inputs[1]].shape[0]
        if len(parent.inputs) > 2:
            bias_input_name = parent.inputs[2]
            bias = parent.input_tensors[bias_input_name]
        else:
            bias_input_name = "{}_bias".format(parent.name,)
            parent.inputs.append(bias_input_name)
            bias = np.zeros(
                (output_channels,), dtype=np.float32
            )
            parent.input_tensors[bias_input_name] = bias
        bias = bias + child.input_tensors[child.inputs[1]]
        parent.input_tensors[bias_input_name] = bias
        parent.outputs = child.outputs
        parent.children.remove(child)
        child.parents.remove(parent)
        return [parent]


class BNBroadcastedMulFuser(NodesFuser):
    '''
    Fuses Mul into BatchNorm
    '''
    def __init__(self):  # type: () -> None
        super(BNBroadcastedMulFuser, self).__init__(2)

    def is_eligible(self, graph, nodes):  # type: (Graph, Sequence[Node]) -> bool
        parent, child = nodes[0], nodes[1]
        if parent.op_type != 'BatchNormalization':
            return False
        if child.op_type != 'Mul':
            return False
        if "broadcast" not in child.attrs:
            return False
        if child.attrs["broadcast"] != 1:
            return False
        if "axis" not in child.attrs:
            return False
        if child.attrs["axis"] != 1:
            return False
        if child.inputs[1] not in child.input_tensors:
            return False
        if parent.inputs[1] not in parent.input_tensors:
            return False
        if parent.inputs[2] not in parent.input_tensors:
            return False
        return True

    def merge(self, graph, nodes):  # type: (Graph, Sequence[Node]) -> Sequence[Node]
        parent, child = nodes[0], nodes[1]
        weight = parent.input_tensors[parent.inputs[1]]
        bias = parent.input_tensors[parent.inputs[2]]
        W = child.input_tensors[child.inputs[1]]
        parent.input_tensors[parent.inputs[1]] = np.multiply(weight, W)
        parent.input_tensors[parent.inputs[2]] = np.multiply(bias, W)
        parent.outputs = child.outputs
        parent.children.remove(child)
        child.parents.remove(parent)
        return [parent]


class BNBroadcastedAddFuser(NodesFuser):
    '''
    Fuses Add into BatchNorm
    '''
    def __init__(self):  # type: () -> None
        super(BNBroadcastedAddFuser, self).__init__(2)

    def is_eligible(self, graph, nodes):  # type: (Graph, Sequence[Node]) -> bool
        parent, child = nodes[0], nodes[1]
        if parent.op_type != 'BatchNormalization':
            return False
        if child.op_type != 'Add':
            return False
        if "broadcast" not in child.attrs:
            return False
        if child.attrs["broadcast"] != 1:
            return False
        if "axis" not in child.attrs:
            return False
        if child.attrs["axis"] != 1:
            return False
        if len(child.inputs) != 2:
            return False
        if child.inputs[1] not in child.input_tensors:
            return False
        if parent.inputs[2] not in parent.input_tensors:
            return False
        return True

    def merge(self, graph, nodes):  # type: (Graph, Sequence[Node]) -> Sequence[Node]
        parent, child = nodes[0], nodes[1]
        bias = parent.input_tensors[parent.inputs[2]]
        b = child.input_tensors[child.inputs[1]]
        parent.input_tensors[parent.inputs[2]] = bias + b
        parent.outputs = child.outputs
        parent.children.remove(child)
        child.parents.remove(parent)
        return [parent]


class DropoutRemover(NodesFuser):
    '''
    Removes Dropout layer
    '''
    def __init__(self):  # type: () -> None
        super(DropoutRemover, self).__init__(2)

    def is_eligible(self, graph, nodes):  # type: (Graph, Sequence[Node]) -> bool
        child = nodes[1]
        return child.op_type == "Dropout"

    def merge(self, graph, nodes):  # type: (Graph, Sequence[Node]) -> Sequence[Node]
        parent, child = nodes[0], nodes[1]
        parent.children.remove(child)
        child.parents.remove(parent)
        parent.outputs = child.outputs
        return [parent]


class ReshapeInitTensorFuser(object):
    '''
    Fuses Reshape operator if it is used only to reshape blob in
    graph initializer. We can reshape here instead of runtime.
    '''

    def __call__(self, graph):  # type: (Graph) -> Graph
        nodes = graph.nodes
        removed = []
        for node in nodes:
            if node.op_type != 'Reshape':
                continue
            if not (len(node.input_tensors) == 2 or len(node.input_tensors) == 1):
                continue
            tensor_name = node.inputs[0]
            if tensor_name not in node.input_tensors:
                continue
            if len(node.inputs) > 1:
                shape_name = node.inputs[1]
                if shape_name not in node.input_tensors:
                    continue
            is_non_constant_parent = False
            if len(node.parents) > 0:
                for parent in node.parents:
                    if parent.op_type != 'Constant':
                        is_non_constant_parent = True
                        break
            if is_non_constant_parent:
                continue

            removed.append(node)
            output_name = node.outputs[0]

            tensor = node.input_tensors[tensor_name]
            if 'shape' in node.attrs:
                shape = tuple(node.attrs["shape"])
            else:
                shape = node.input_tensors[shape_name] # type: ignore

            # ONNX spec supports setting dimension to '0', in which case
            # it should be taken from old dimension.
            # This isn't supported in numpy, so don't transform.
            # TODO Should we support this case?
            if any([s == 0 for s in shape]):
                continue

            reshaped_tensor = tensor.reshape(shape)

            for child in node.children:
                child.parents.remove(node)
                child.input_tensors[output_name] = reshaped_tensor

        transformed_nodes = [node for node in nodes if node not in removed]
        return Graph(transformed_nodes, graph.inputs, graph.outputs, graph.shape_dict)


class OutputRenamer(object):
    '''
    Rename outputs according to mapping
    '''
    def __init__(self,
                 mapping,  # type: Dict[Text, Text]
                 ):
        # type: (...) -> None
        self.mapping = mapping

    def __call__(self, graph):  # type: (Graph) -> Graph
        mapping = self.mapping.copy()
        nodes = graph.nodes
        for node in nodes:
            for i in range(len(node.outputs)):
                output = node.outputs[i]
                if output not in mapping:
                    continue
                node.outputs[i] = mapping[output]
                for child in node.children:
                    for j in range(len(child.inputs)):
                        input_ = child.inputs[j]
                        if input_ != output:
                            continue
                        child.inputs[j] = mapping[output]
                del mapping[output]
                if len(mapping) == 0:
                    break
        return graph


class PixelShuffleFuser(NodesFuser):
    '''
    Fuses 3 operators reshape->transpose->reshape which is equivalent to
    pytorch's pixel_shuffle layer
    '''
    def __init__(self):  # type: () -> None
        super(PixelShuffleFuser, self).__init__(3)
        self.num_added = 0

    def is_eligible(self, graph, nodes):  # type: (Graph, Sequence[Node]) -> bool
        if nodes[0].op_type != 'Reshape':
            return False
        if nodes[1].op_type != 'Transpose':
            return False
        if nodes[2].op_type != 'Reshape':
            return False
        if nodes[0].inputs[1] not in nodes[0].input_tensors:
            return False
        if nodes[2].inputs[1] not in nodes[2].input_tensors:
            return False

        shape = nodes[0].input_tensors[nodes[0].inputs[1]]
        if len(shape) != 6:
            return False
        if shape[0] != 1 or shape[2] != shape[3]:
            return False

        input_channels = shape[1]
        scale_factor = shape[2]
        input_height = shape[4]
        input_width = shape[5]

        if nodes[1].attrs.get('perm', []) != [0, 1, 4, 2, 5, 3]:
            return False

        shape = nodes[2].input_tensors[nodes[2].inputs[1]]
        if len(shape) != 4:
            return False

        output_channels = shape[1]
        output_height = shape[2]
        output_width = shape[3]
        if input_channels != output_channels:
            return False
        if (input_height * scale_factor) != output_height:
            return False
        if (input_width * scale_factor) != output_width:
            return False

        return True

    def get_unique_edge_name(self, graph, name):  # type: (Graph, Text) -> Text
        self.num_added += 1
        return graph.get_unique_edge_name(name + '_' + str(self.num_added))

    def merge(self, graph, nodes):  # type: (Graph, Sequence[Node]) -> Sequence[Node]
        '''
        Pixel shuffle is implemented using 3 operators:
            - Reshape(1, channels, scale, scale, height, width)
            - Transpose(0, 1, 4, 2, 5, 3)
            - Reshape(1, channels, height * scale, width * scale)
        CoreML Reshape and Transpose layers don't support tensors with more
        than 4 dimensions. Thus we change above sequence of operators to the
        following equivalent sequence:
            - Reshape(channels, scale * scale, height, width)
            - Transpose(0, 2, 1, 3)
            - Reshape(channels * height, scale, scale, width)
            - Transpose(0, 1, 3, 2)
            - Reshape(1, channels, height * scale, width * scale)
        '''
        reshape_1 = nodes[0]
        transpose_1 = nodes[1]
        transpose_1.children = []

        shape = reshape_1.input_tensors[reshape_1.inputs[1]]

        channels = shape[1]
        scale = shape[2]
        height = shape[4]
        width = shape[5]

        reshape_1.input_tensors[reshape_1.inputs[1]] = np.asarray([channels, scale * scale, height, width])
        transpose_1.attrs['perm'] = [0, 2, 1, 3]

        reshape_output_name = 'pixel_shuffle_reshape'
        transpose_output_name = 'pixel_shuffle_transpose'

        transpose_1.outputs = [
            self.get_unique_edge_name(graph, transpose_output_name)
        ]

        shape_name_second_reshape = self.get_unique_edge_name(graph, reshape_output_name)
        output_name_second_reshape = self.get_unique_edge_name(graph, reshape_output_name)
        reshape_2 = Node(
            reshape_output_name,
            'Reshape',
            {},
            [transpose_1.outputs[0], shape_name_second_reshape],
            [output_name_second_reshape]
        )
        reshape_2.input_tensors[shape_name_second_reshape] = np.asarray([channels * height, scale, scale, width])
        transpose_1.add_child(reshape_2)

        transpose_2 = Node(
            transpose_output_name,
            'Transpose',
            {'perm': [0, 1, 3, 2]},
            reshape_2.outputs,
            [self.get_unique_edge_name(graph, transpose_output_name)]
        )
        reshape_2.add_child(transpose_2)

        final_reshape = nodes[2]
        final_reshape.inputs = [transpose_2.outputs[0], nodes[2].inputs[1]]
        final_reshape.parents = []
        transpose_2.add_child(final_reshape)
        return [reshape_1, transpose_1, reshape_2, transpose_2, final_reshape]


class AddModelInputsOutputs(object):
    '''
    Expose hidden states of recurrent layers as model inputs and outputs
    '''
    def __call__(self, graph):  # type: (Graph) -> Graph
        input_names = [str(input_[0]) for input_ in graph.inputs]
        output_names = [str(output_[0]) for output_ in graph.outputs]
        for node in graph.nodes:
            if str(node.op_type) == 'LSTM':
                input_h = node.inputs[5] if len(node.inputs) > 5 else node.inputs[0] + '_h_input'
                input_c = node.inputs[6] if len(node.inputs) > 6 else node.inputs[0] + '_c_input'
                output_h = node.outputs[1] if len(node.outputs) > 1 else node.outputs[0] + '_h_output'
                output_c = node.outputs[2] if len(node.outputs) > 2 else node.outputs[0] + '_c_output'
                h = node.attrs["hidden_size"]
                for input_ in [str(input_h), str(input_c)]:
                    if input_ not in input_names:
                        graph.inputs.append(tuple((input_, TensorProto.FLOAT, (h,))))  #type: ignore
                    if input_ not in graph.blob_to_op_type:
                        graph.blob_to_op_type[input_] = ['LSTM']
                for output_ in [str(output_h), str(output_c)]:
                    if output_ not in output_names:
                        graph.outputs.append(tuple((output_, TensorProto.FLOAT, (h,))))  #type: ignore
                    graph.blob_from_op_type[output_] = 'LSTM'
        return graph


class ConstantsToInitializers(object):
    '''
    Takes onnx Constant nodes and puts the tensor into graph initializers instead.
    '''
    def __call__(self, graph):  # type: (Graph) -> Graph
        output_names = [str(output_[0]) for output_ in graph.outputs]
        remaining_nodes = []
        for node in graph.nodes:
            if node.op_type != 'Constant' or node.name in output_names:
                remaining_nodes.append(node)
                continue
            for child in node.children:
                child.input_tensors[node.outputs[0]] = node.attrs["value"]

        graph.nodes = remaining_nodes
        return graph


class ImageScalerRemover(object):
    '''
    Removes ImageScaler layer if connected to a model input and single parent child nodes
    '''

    def __call__(self, graph):  # type: (Graph) -> Graph
        input_names = [str(input_[0]) for input_ in graph.inputs]
        nodes_to_be_removed = []
        for node in graph.nodes:
            if (node.op_type != 'ImageScaler') or (len(node.parents) != 0) or (node.inputs[0] not in input_names):
                continue
            is_eligible = True
            for child in node.children:
                if not (len(child.parents) == 1 and child.inputs[0] == node.outputs[0]):
                    is_eligible = False
                    break
                child.inputs[0] = node.inputs[0]
                child.parents = []
            if not is_eligible:
                continue
            nodes_to_be_removed.append(node.name)

        transformed_nodes = []
        for node in graph.nodes:
            if node.name not in nodes_to_be_removed:
                transformed_nodes.append(node)
        return Graph(transformed_nodes, graph.inputs, graph.outputs, graph.shape_dict)