from typing import List

import torch
from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SISDR

from src.base.base_metric import BaseMetric


class SISDRMetric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.sisdr = SISDR().to(device)

    def __call__(self, pred, target, *args, **kwargs):
        valid_size = min(pred.size(-1), target.size(-1))
        sisdrs = [
            self.sisdr(
                pred[i, 0, :valid_size].to(self.device),
                target[i, 0, :valid_size].to(self.device),
            )
            for i in range(pred.size(0))
        ]
        return sum(sisdrs) / len(sisdrs)
