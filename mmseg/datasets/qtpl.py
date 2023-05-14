# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class QTPLDataset(CustomDataset):
    CLASSES = ('background', 'lake')
    PALETTE = [[0, 0, 0], [128, 0, 0]]

    def __init__(self, **kwargs):
        super(QTPLDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)
