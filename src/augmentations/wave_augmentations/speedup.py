import torch_audiomentations
import torch
import librosa

from src.augmentations.base import AugmentationBase


class Speedup(AugmentationBase):
    def __init__(self, speedup_min: float, speedup_max: float, *args, **kwargs):
        self._speedup_min = speedup_min
        self._speedup_max = speedup_max

    def __call__(self, data: torch.Tensor):
        speedup = self._speedup_min + torch.rand(1).item() * (
            self._speedup_max - self._speedup_min
        )
        return torch.from_numpy(
            librosa.effects.time_stretch(data.numpy(), rate=speedup)
        )
