# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='APDEncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SiaResAPD_18',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='Seg_head',
        in_channels=[256, 256, 256, 64, 2, 2],
        in_index=[0, 1, 2, 3, 4, 5],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='ContrastiveLoss', use_sigmoid=False, loss_weight=1.0)]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))