dataset_type = 'LandCoverAi'
data_root = '/home/ilya/PycharmProjects/mrsis/intership/git_inter/task_3/images/'

# crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromGeoTiff'),
    dict(type='LoadAnnotationsGeoTiff', reduce_zero_label=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    #dict(type='RandomCrop', crop_size=crop_size),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromGeoTiff'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotationsGeoTiff', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

tta_pipeline = [
    dict(type='LoadImageFromGeoTiff'),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale_factor=ratio, keep_ratio=True) for ratio in [0.5, 0.75, 1.0, 1.25]],
           # [dict(type='RandomCrop', crop_size=crop_size)],
            [dict(type='PackSegInputs')]
        ]
    )
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        data_prefix=dict(
            img_path='/home/ilya/PycharmProjects/mrsis/intership/git_inter/task_3/images',
            seg_map_path='/home/ilya/PycharmProjects/mrsis/intership/git_inter/task_3/masks'
        ),
        ann_file="/home/ilya/PycharmProjects/mrsis/intership/git_inter/task_3/train.txt",
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        data_prefix=dict(
            img_path='/home/ilya/PycharmProjects/mrsis/intership/git_inter/task_3/images',
            seg_map_path='/home/ilya/PycharmProjects/mrsis/intership/git_inter/task_3/masks'
        ),
        ann_file="/home/ilya/PycharmProjects/mrsis/intership/git_inter/task_3/test.txt",
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator