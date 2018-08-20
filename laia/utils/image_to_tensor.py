from __future__ import division

import warnings

from laia.data.transforms.vision import ToPreparedTensor


# TODO: Remove this
class ImageToTensor(ToPreparedTensor):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The use of laia.utils.ImageToTensor is deprecated, "
            "please use laia.data.transforms.vision.ToPreparedTensor instead."
        )
        super(ImageToTensor, self).__init__(*args, **kwargs)
