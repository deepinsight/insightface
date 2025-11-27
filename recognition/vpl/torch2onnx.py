import numpy as np
import onnx
import torch


def convert_onnx(net, path_module, output, opset=11, simplify=False):
    assert isinstance(net, torch.nn.Module)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(np.float32)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()

    weight = torch.load(path_module)
    net.load_state_dict(weight)
    net.eval()
    torch.onnx.export(net, img, output, keep_initializers_as_inputs=False, verbose=False, opset_version=opset)
    model = onnx.load(output)
    graph = model.graph
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output)

    
if __name__ == '__main__':
    import os
    import argparse
    from backbones import get_model

    parser = argparse.ArgumentParser(description='ArcFace PyTorch to onnx')
    parser.add_argument('input', type=str, help='input backbone.pth file or path')
    parser.add_argument('--output', type=str, default=None, help='output onnx path')
    parser.add_argument('--network', type=str, default=None, help='backbone network')
    parser.add_argument('--simplify', type=bool, default=True, help='onnx simplify')
    args = parser.parse_args()
    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "backbone.pth")
    assert os.path.exists(input_file)
    model_name = os.path.basename(os.path.dirname(input_file)).lower()
    params = model_name.split("_")
    if len(params) >= 3 and params[1] in ('arcface', 'cosface'):
        if args.network is None:
            args.network = params[2]
    assert args.network is not None
    print(args)
    backbone_onnx = get_model(args.network, dropout=0)

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), 'onnx')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    assert os.path.isdir(output_path)
    output_file = os.path.join(output_path, "%s.onnx" % model_name)
    convert_onnx(backbone_onnx, input_file, output_file, simplify=args.simplify)
