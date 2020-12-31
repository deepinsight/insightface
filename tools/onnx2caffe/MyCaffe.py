from collections import OrderedDict, Counter

from caffe.proto import caffe_pb2
from google import protobuf
import six

def param_name_dict():
    """Find out the correspondence between layer names and parameter names."""

    layer = caffe_pb2.LayerParameter()
    # get all parameter names (typically underscore case) and corresponding
    # type names (typically camel case), which contain the layer names
    # (note that not all parameters correspond to layers, but we'll ignore that)
    param_names = [f.name for f in layer.DESCRIPTOR.fields if f.name.endswith('_param')]
    param_type_names = [type(getattr(layer, s)).__name__ for s in param_names]
    # strip the final '_param' or 'Parameter'
    param_names = [s[:-len('_param')] for s in param_names]
    param_type_names = [s[:-len('Parameter')] for s in param_type_names]
    return dict(zip(param_type_names, param_names))

def assign_proto(proto, name, val):
    """Assign a Python object to a protobuf message, based on the Python
    type (in recursive fashion). Lists become repeated fields/messages, dicts
    become messages, and other types are assigned directly. For convenience,
    repeated fields whose values are not lists are converted to single-element
    lists; e.g., `my_repeated_int_field=3` is converted to
    `my_repeated_int_field=[3]`."""

    is_repeated_field = hasattr(getattr(proto, name), 'extend')
    if is_repeated_field and not isinstance(val, list):
        val = [val]
    if isinstance(val, list):
        if isinstance(val[0], dict):
            for item in val:
                proto_item = getattr(proto, name).add()
                for k, v in six.iteritems(item):
                    assign_proto(proto_item, k, v)
        else:
            getattr(proto, name).extend(val)
    elif isinstance(val, dict):
        for k, v in six.iteritems(val):
            assign_proto(getattr(proto, name), k, v)
    else:
        setattr(proto, name, val)

class Function(object):
    """A Function specifies a layer, its parameters, and its inputs (which
    are Tops from other layers)."""

    def __init__(self, type_name, layer_name, inputs,outputs, **params):
        self.type_name = type_name
        self.inputs = inputs
        self.outputs = outputs
        self.params = params
        self.layer_name = layer_name
        self.ntop = self.params.get('ntop', 1)
        # use del to make sure kwargs are not double-processed as layer params
        if 'ntop' in self.params:
            del self.params['ntop']
        self.in_place = self.params.get('in_place', False)
        if 'in_place' in self.params:
            del self.params['in_place']
        # self.tops = tuple(Top(self, n) for n in range(self.ntop))l

    def _get_name(self, names, autonames):
        if self not in names and self.ntop > 0:
            names[self] = self._get_top_name(self.tops[0], names, autonames)
        elif self not in names:
            autonames[self.type_name] += 1
            names[self] = self.type_name + str(autonames[self.type_name])
        return names[self]

    def _get_top_name(self, top, names, autonames):
        if top not in names:
            autonames[top.fn.type_name] += 1
            names[top] = top.fn.type_name + str(autonames[top.fn.type_name])
        return names[top]

    def _to_proto(self):
        bottom_names = []
        for inp in self.inputs:
            # inp._to_proto(layers, names, autonames)
            bottom_names.append(inp)
        layer = caffe_pb2.LayerParameter()
        layer.type = self.type_name
        layer.bottom.extend(bottom_names)

        if self.in_place:
            layer.top.extend(layer.bottom)
        else:
            for top in self.outputs:
                layer.top.append(top)
        layer.name = self.layer_name
        # print(self.type_name + "...")
        for k, v in six.iteritems(self.params):
            # special case to handle generic *params
            # print("generating "+k+"...")

            if k.endswith('param'):
                assign_proto(layer, k, v)
            else:
                try:
                    assign_proto(getattr(layer,
                        _param_names[self.type_name] + '_param'), k, v)
                except (AttributeError, KeyError):
                    assign_proto(layer, k, v)

        return layer

class Layers(object):
    """A Layers object is a pseudo-module which generates functions that specify
    layers; e.g., Layers().Convolution(bottom, kernel_size=3) will produce a Top
    specifying a 3x3 convolution applied to bottom."""

    def __getattr__(self, name):
        def layer_fn(*args, **kwargs):
            fn = Function(name, args, kwargs)
            return fn
        return layer_fn




_param_names = param_name_dict()

