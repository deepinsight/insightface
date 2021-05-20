import argparse
import os
import os.path as osp
import pickle
import numpy as np
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.core.evaluation import wider_evaluation, get_widerface_gts


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default='wout', help='output folder')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--save-preds', action='store_true', help='save results')
    parser.add_argument('--show-assign', action='store_true', help='show bbox assign')
    parser.add_argument('--debug', action='store_true', help='debug flag')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--thr',
        type=float,
        default=0.02,
        help='score threshold')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()


    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    gt_path = os.path.join(os.path.dirname(cfg.data.test.ann_file), 'gt')
    pipelines = cfg.data.test.pipeline
    for pipeline in pipelines:
        if pipeline.type=='MultiScaleFlipAug':
            if args.mode==0: #640 scale
                pipeline.img_scale = (640, 640)
            elif args.mode==1: #for single scale in other pages
                pipeline.img_scale = (1100, 1650)
            elif args.mode==2: #original scale
                pipeline.img_scale = None
                pipeline.scale_factor = 1.0
            elif args.mode>30:
                pipeline.img_scale = (args.mode, args.mode)
            transforms = pipeline.transforms
            for transform in transforms:
                if transform.type=='Pad':
                    if args.mode!=2:
                        transform.size = pipeline.img_scale
                    else:
                        transform.size = None
                        transform.size_divisor = 32
    print(cfg.data.test.pipeline)
    distributed = False

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    cfg.test_cfg.score_thr = args.thr

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if args.show_assign:
        gts_easy, gts_medium, gts_hard = get_widerface_gts(gt_path)
        assign_stat = [0, 0]
        gts_size = []
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    results = {}
    output_folder = args.out
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        assert len(result)==1
        batch_size = 1
        result = result[0][0]
        img_metas = data['img_metas'][0].data[0][0]
        filepath = img_metas['ori_filename']
        det_scale = img_metas['scale_factor'][0]
        #print(img_metas)
        ori_shape = img_metas['ori_shape']
        img_width = ori_shape[1]
        img_height = ori_shape[0]
        _vec = filepath.split('/')
        pa, pb = _vec[-2], _vec[1]
        if pa not in results:
            results[pa] = {}
        xywh = result.copy()
        w = xywh[:,2] - xywh[:,0]
        h = xywh[:,3] - xywh[:,1]
        xywh[:,2] = w
        xywh[:,3] = h

        event_name = pa
        img_name = pb.rstrip('.jpg')
        results[event_name][img_name] = xywh
        if args.save_preds:
            out_dir = os.path.join(output_folder, pa)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_file = os.path.join(out_dir, pb.replace('jpg', 'txt'))
            boxes = result
            with open(out_file, 'w') as f:
                name = '/'.join([pa, pb])
                f.write("%s\n"%(name))
                f.write("%d\n"%(boxes.shape[0]))
                for b in range(boxes.shape[0]):
                    box = boxes[b]
                    f.write("%.5f %.5f %.5f %.5f %g\n"%(box[0], box[1], box[2]-box[0], box[3]-box[1], box[4]))

        if args.show_assign:
            assert args.mode==0
            input_height, input_width = 640, 640
            gt_hard = gts_hard[event_name][img_name]
            #print(event_name, img_name, gt_hard.shape)
            gt_bboxes = gt_hard * det_scale
            bbox_width = gt_bboxes[:,2] - gt_bboxes[:,0]
            bbox_height = gt_bboxes[:,3] - gt_bboxes[:,1]
            bbox_area = bbox_width * bbox_height
            gt_size = np.sqrt(bbox_area+0.0001)
            gts_size += list(gt_size)
            anchor_cxs = []
            anchor_cys = []
            for idx, stride in enumerate([8,16,32,64,128]):
                height = input_height // stride
                width = input_width // stride
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                anchor_cx = anchor_centers[:,0]
                anchor_cy = anchor_centers[:,1]
                anchor_cxs += list(anchor_cx)
                anchor_cys += list(anchor_cy)
            anchor_cx = np.array(anchor_cxs, dtype=np.float32)
            anchor_cy = np.array(anchor_cys, dtype=np.float32)

            num_gts = gt_bboxes.shape[0]
            num_anchors = anchor_cx.shape[0]
            anchor_cx = np.broadcast_to(anchor_cx.reshape((1,-1)), (num_gts, num_anchors)).reshape(num_anchors, num_gts)
            anchor_cy = np.broadcast_to(anchor_cy.reshape((1,-1)), (num_gts, num_anchors)).reshape(num_anchors, num_gts)
            gt_x1 = gt_bboxes[:,0]
            gt_y1 = gt_bboxes[:,1]
            gt_x2 = gt_bboxes[:,2]
            gt_y2 = gt_bboxes[:,3]
            gt_cover = np.zeros( (gt_bboxes.shape[0], ), dtype=np.float32)
            l_ = anchor_cx - gt_x1
            t_ = anchor_cy - gt_y1
            r_ = gt_x2 - anchor_cx
            b_ = gt_y2 - anchor_cy
            dist = np.stack([l_, t_, r_, b_], axis=1).min(axis=1)
            gt_dist = dist.max(axis=0)
            gt_dist  = gt_dist / gt_size
            center_thres = 0.01
            #center_thres = -0.25
            gt_cover_inds = np.where(gt_dist>center_thres)[0]
            num_assigned = len(gt_cover_inds)
            assign_stat[0] += num_gts
            assign_stat[1] += num_assigned
            




        for _ in range(batch_size):
            prog_bar.update()
    aps = wider_evaluation(results, gt_path, 0.5, args.debug)
    with open(os.path.join(output_folder, 'aps'), 'w') as f:
        f.write("%f,%f,%f\n"%(aps[0],aps[1],aps[2]))
    print('APS:', aps)
    if args.show_assign:
        print('ASSIGN:', assign_stat)
        gts_size = np.array(gts_size, dtype=np.float32)
        gts_size = np.sort(gts_size)
        assert len(gts_size)==assign_stat[0]
        print(gts_size[assign_stat[0]//2])


if __name__ == '__main__':
    main()
