_base_ = [
    '../_base_/models/isanet_r50-d8.py', '../_base_/datasets/land_cover_ai.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_2k_custom.py'
]
#original 512 1024
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
