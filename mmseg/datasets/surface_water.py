from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SurfaceWaterDataset(CustomDataset):
    """ISPRS Potsdam dataset.

        In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
        ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
        ``seg_map_suffix`` are both fixed to '.png'.
        """
    CLASSES = ('background', 'lake')
    PALETTE = [[0, 0, 0], [128, 0, 0]]

    def __init__(self, **kwargs):
        super(SurfaceWaterDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)
