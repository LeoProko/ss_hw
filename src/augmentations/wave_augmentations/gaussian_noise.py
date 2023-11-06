import torch_audiomentations
import torch

from src.augmentations.base import AugmentationBase


class GaussianNoise(AugmentationBase):
    def __init__(self, variance: float, *args, **kwargs):
        self._noiser = torch.distributions.Normal(0, variance)

    def __call__(self, data: torch.Tensor):
        return data + self._noiser.sample(data.size())
