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
    dict(type='RandomSquareCrop',
         crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0]),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=False),
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
        img_scale=(1100, 1650),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32, pad_val=0),
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
#evaluation = dict(interval=1, metric='bbox')
evaluation = dict(interval=10, metric='mAP')
