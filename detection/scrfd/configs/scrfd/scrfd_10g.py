optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
lr_mult = 8
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.001,
    step=[55*lr_mult, 68*lr_mult])
total_epochs = 80*lr_mult
checkpoint_config = dict(interval=80)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
dataset_type = 'RetinaFaceDataset'
data_root = 'data/retinaface/'
train_root = 'data/retinaface/train/'
val_root = 'data/retinaface/val/'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
    dict(
        type='RandomSquareCrop',
        crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        bbox_clip_border=False),
    dict(
        type='Resize',
        img_scale=(640, 640),
        keep_ratio=False,
        bbox_clip_border=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[128.0, 128.0, 128.0],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore',
            'gt_keypointss'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128.0, 128.0, 128.0],
                to_rgb=True),
            dict(type='Pad', size=(640, 640), pad_val=0),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=3,
    train=dict(
        type='RetinaFaceDataset',
        ann_file='data/retinaface/train/labelv2.txt',
        img_prefix='data/retinaface/train/images/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
            dict(
                type='RandomSquareCrop',
                crop_choice=[
                    0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0
                ],
                bbox_clip_border=False),
            dict(
                type='Resize',
                img_scale=(640, 640),
                keep_ratio=False,
                bbox_clip_border=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128.0, 128.0, 128.0],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore',
                    'gt_keypointss'
                ])
        ]),
    val=dict(
        type='RetinaFaceDataset',
        ann_file='data/retinaface/val/labelv2.txt',
        img_prefix='data/retinaface/val/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[128.0, 128.0, 128.0],
                        to_rgb=True),
                    dict(type='Pad', size=(640, 640), pad_val=0),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='RetinaFaceDataset',
        ann_file='data/retinaface/val/labelv2.txt',
        img_prefix='data/retinaface/val/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[128.0, 128.0, 128.0],
                        to_rgb=True),
                    dict(type='Pad', size=(640, 640), pad_val=0),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
model = dict(
    type='SCRFD',
    backbone=dict(
        type='ResNetV1e',
        depth=0,
        block_cfg=dict(
            block='BasicBlock',
            stage_blocks=(3, 4, 2, 3),
            stage_planes=[56, 88, 88, 224]),
        base_channels=56,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='PAFPN',
        in_channels=[56, 88, 88, 224],
        out_channels=56,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=3),
    bbox_head=dict(
        type='SCRFDHead',
        num_classes=1,
        in_channels=56,
        stacked_convs=3,
        feat_channels=80,
        #norm_cfg=dict(type='BN', requires_grad=True),
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True),
        cls_reg_share=True,
        strides_share=True,
        scale_mode=2,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales = [1,2],
            base_sizes = [16, 64, 256],
            strides=[8, 16, 32]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=False,
        reg_max=8,
        loss_bbox=dict(type='DIoULoss', loss_weight=2.0),
        use_kps=False,
        loss_kps=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=0.1),
        train_cfg=dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=-1,
            min_bbox_size=0,
            score_thr=0.02,
            nms=dict(type='nms', iou_threshold=0.45),
            max_per_img=-1)))
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=-1,
    min_bbox_size=0,
    score_thr=0.02,
    nms=dict(type='nms', iou_threshold=0.45),
    max_per_img=-1)
epoch_multi = 1
evaluation = dict(interval=80, metric='mAP')
