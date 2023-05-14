_base_ = [
    '../_base_/models/setr_naive.py', '../../local_configs/_base_/datasets/surface_water_256x256.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained=None,
    backbone=dict(
        img_size=(256, 256),
        drop_rate=0.,
        init_cfg=dict(
            type='Pretrained', checkpoint='/gpfs/home/chenben/models/SETR/jx_vit_large_p16_384-b3be5167.pth')),
    decode_head=dict(num_classes=2),
    auxiliary_head=[
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=0,
            num_classes=150,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=1,
            num_classes=150,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=2,
            num_classes=150,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    # test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(341, 341)),
test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(150, 150)),
)

optimizer = dict(
    lr=0.01,
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))

# num_gpus: 8 -> batch_size: 16
data = dict(samples_per_gpu=16)
