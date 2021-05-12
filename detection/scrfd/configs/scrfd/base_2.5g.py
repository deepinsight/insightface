_base_ = [
    #'../_base_/datasets/retinaface.py',
    '../_base_/schedules/schedule_retinaface_sgd.py', '../_base_/default_runtime.py'
]
dataset_type = 'RetinaFaceDataset'
data_root = 'data/retinaface/'
train_root = data_root+'train/'
val_root = data_root+'val/'
#img_norm_cfg = dict(
#    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0], to_rgb=True)
train_pipeline = [
    #dict(type='LoadImageFromFile'),
    #dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    #dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size_divisor=32),
    #dict(type='DefaultFormatBundle'),
    #dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_keypoints', 'gt_labels']),

    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
    dict(type='RandomSquareCrop', crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]),
    #dict(type='RandomSquareCrop', crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0]),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_keypointss']),
    #dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),
]
test_pipeline = [
    #dict(type='LoadImageFromFile'),
    #dict(
    #    type='MultiScaleFlipAug',
    #    img_scale=(1333, 800),
    #    flip=False,
    #    transforms=[
    #        dict(type='Resize', keep_ratio=True),
    #        dict(type='RandomFlip'),
    #        dict(type='Normalize', **img_norm_cfg),
    #        dict(type='Pad', size_divisor=32),
    #        dict(type='ImageToTensor', keys=['img']),
    #        dict(type='Collect', keys=['img']),
    #    ])

    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        #img_scale=(1100, 1650),
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            #dict(type='Pad', size_divisor=32, pad_val=0),
            dict(type='Pad', size=(640,640), pad_val=0),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=train_root + 'labelv2.txt',
        #ann_file=train_root + 'label_wo.txt',
        img_prefix=train_root+ 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=val_root + 'labelv2.txt',
        img_prefix=val_root+ 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=val_root + 'labelv2.txt',
        img_prefix=val_root+ 'images/',
        pipeline=test_pipeline),
    )
model = dict(
    type='SCRFD',
    #pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNetV1e',
        depth=34,
        base_channels=16,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        #frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        #norm_eval=True,
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='PAFPN',
        #in_channels=[64, 128, 256, 512],
        in_channels=[16, 32, 64, 128],
        out_channels=48,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=3),
    bbox_head=dict(
        type='SCRFDHead',
        num_classes=1,
        in_channels=48,
        stacked_convs=2,
        feat_channels=96,
        #norm_cfg=dict(type='BN', requires_grad=True),
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True),
        cls_reg_share = True,
        strides_share = True,
        scale_mode = 2,
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
        #loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        loss_dfl=False,
        reg_max=8,
        #loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_bbox=dict(type='DIoULoss', loss_weight=2.0),
        use_kps = False,
        loss_kps=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.1),
        )
    )
# training and testing settings
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    #assigner=dict(
    #    type='MaxIoUAssigner',
    #    pos_iou_thr=0.5,
    #    neg_iou_thr=0.3,
    #    min_pos_iou=0.5,
    #    ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    #nms_pre=1000,
    nms_pre=-1,
    min_bbox_size=0,
    score_thr=0.02,
    nms=dict(type='nms', iou_threshold=0.45),
    max_per_img=-1)
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005)
epoch_multi = 8
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.001,
    step=[55*epoch_multi, 68*epoch_multi])
total_epochs = 80*epoch_multi
#checkpoint_config = dict(interval=1)
checkpoint_config = dict(interval=80)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
evaluation = dict(interval=80, metric='mAP')




