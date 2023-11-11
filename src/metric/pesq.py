from typing import List

import torch
from torch import Tensor
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ

from src.base.base_metric import BaseMetric
from src.base.base_text_encoder import BaseTextEncoder
from src.metric.utils import calc_cer


class PESQMetric(BaseMetric):
    def __init__(self, sr: int, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.pesq = PESQ(sr, "wb").to(device)

    def __call__(self, pred, target, *args, **kwargs):
        valid_size = min(pred.size(-1), target.size(-1))
        pesqs = [
            self.pesq(pred[i, 0, :valid_size].to(self.device), target[i, 0, :valid_size].to(self.device))
            for i in range(pred.size(0))
            if not (target[i, 0, :valid_size] == 0).all() and not (pred[i, 0, :valid_size] == 0).all()
        ]
        return sum(pesqs) / len(pesqs) if pesqs else 0
