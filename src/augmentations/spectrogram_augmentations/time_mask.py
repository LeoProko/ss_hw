import torchaudio
from torch import Tensor

from src.augmentations.base import AugmentationBase


class TimeMask(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchaudio.transforms.TimeMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        return self._aug(data)
