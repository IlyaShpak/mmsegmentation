# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseGifDataset


@DATASETS.register_module()
class LandCoverAi(BaseGifDataset):
    """Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('building', 'woodland', 'water', 'road'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
