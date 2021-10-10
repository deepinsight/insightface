import argparse
import os.path as osp

import numpy as np
import onnx
import os
#import onnxruntime as rt
import torch

from mmdet.core import (build_model_from_cfg, generate_inputs_and_wrap_model,
                        preprocess_example_input)

#from mmdet.models import build

def pytorch2onnx(config_path,
                 checkpoint_path,
                 input_img,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 simplify = True,
                 dynamic = True,
                 normalize_cfg=None,
                 dataset='coco',
                 test_img=None):

    input_config = {
        'input_shape': input_shape,
        'input_path': input_img,
        'normalize_cfg': normalize_cfg
    }
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    tmp_ckpt_file = None
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
        tmp_ckpt_file = checkpoint_path+"_slim.pth"
        torch.save(checkpoint, tmp_ckpt_file)
        print('remove optimizer params and save to', tmp_ckpt_file)
        checkpoint_path = tmp_ckpt_file

    model, tensor_data = generate_inputs_and_wrap_model(
        config_path, checkpoint_path, input_config)

    if tmp_ckpt_file is not None:
        os.remove(tmp_ckpt_file)

    if simplify or dynamic:
        ori_output_file = output_file.split('.')[0]+"_ori.onnx"
    else:
        ori_output_file = output_file

    # Define input and outputs names, which are required to properly define
    # dynamic axes
    input_names = ['input.1']
    output_names = ['score_8', 'score_16', 'score_32',
                    'bbox_8', 'bbox_16', 'bbox_32',
                    ]

    # If model graph contains keypoints strides add keypoints to outputs
    if 'stride_kps' in str(model):
        output_names += ['kps_8', 'kps_16', 'kps_32']

    # Define dynamic axes for export
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {out: {0: '?', 1: '?'} for out in output_names}
        dynamic_axes[input_names[0]] = {
            0: '?',
            2: '?',
            3: '?'
        }

    torch.onnx.export(
        model,
        tensor_data,
        ori_output_file,
        keep_initializers_as_inputs=False,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version)

    if simplify or dynamic:
        model = onnx.load(ori_output_file)
        if simplify:
            from onnxsim import simplify
            #print(model.graph.input[0])
            if dynamic:
                input_shapes = {model.graph.input[0].name : list(input_shape)}
                model, check = simplify(model, input_shapes=input_shapes, dynamic_input_shape=True)
            else:
                model, check = simplify(model)
            assert check, "Simplified ONNX model could not be validated"
        onnx.save(model, output_file)
        os.remove(ori_output_file)


    print(f'Successfully exported ONNX model: {output_file}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input-img', type=str, help='Images for input')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--test-img', type=str, default=None, help='Images for test')
    parser.add_argument(
        '--dataset', type=str, default='coco', help='Dataset name')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        #default=[640, 640],
        #default=[384, 384],
        default=[-1, -1],
        help='input image size')
    parser.add_argument(
        '--mean',
        type=float,
        nargs='+',
        default=[127.5, 127.5, 127.5],
        help='mean value used for preprocess input data')
    parser.add_argument(
        '--std',
        type=float,
        nargs='+',
        default=[128.0, 128.0, 128.0],
        help='variance value used for preprocess input data')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 11, 'MMDet only support opset 11 now'

    if not args.input_img:
        args.input_img = osp.join(
            osp.dirname(__file__), '../tests/data/t1.jpg')

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    assert len(args.mean) == 3
    assert len(args.std) == 3

    simplify = True
    dynamic = False
    if input_shape[2]<=0 or input_shape[3]<=0:
        input_shape = (1,3,640,640)
        dynamic = True
        #simplify = False
        print('set to dynamic input with dummy shape:', input_shape)

    normalize_cfg = {'mean': args.mean, 'std': args.std}

    if len(args.output_file)==0:
        output_dir = osp.join(osp.dirname(__file__), '../onnx')
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        cfg_name = args.config.split('/')[-1]
        pos = cfg_name.rfind('.')
        cfg_name = cfg_name[:pos]
        if dynamic:
            args.output_file = osp.join(output_dir, "%s.onnx"%cfg_name)
        else:
            args.output_file = osp.join(output_dir, "%s_shape%dx%d.onnx"%(cfg_name,input_shape[2],input_shape[3]))

    # convert model to onnx file
    pytorch2onnx(
        args.config,
        args.checkpoint,
        args.input_img,
        input_shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        simplify = simplify,
        dynamic = dynamic,
        normalize_cfg=normalize_cfg,
        dataset=args.dataset,
        test_img=args.test_img)
