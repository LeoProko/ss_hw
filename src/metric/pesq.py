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

    def __call__(self, pred, target, length, *args, **kwargs):
        lengths = [min(min(length[i], pred.size(-1)), target.size(-1)) for i in range(len(length))]

        pesqs = [
            self.pesq(pred[i, 0, :lengths[i]].to(self.device), target[i, 0, :lengths[i]].to(self.device))
            for i in range(pred.size(0))
            if not ( torch.isclose(target[i, 0, :lengths[i]].mean(), torch.tensor([0.0])) and (target[i, 0, :lengths[i]] > -1e-8).all()
                 or
                torch.isclose(pred[i, 0, :lengths[i]].mean(), torch.tensor([0.0])) and (pred[i, 0, :lengths[i]] > -1e-8).all()
                or (target[i, 0, :lengths[i]] == 0).all()
                or (pred[i, 0, :lengths[i]] == 0).all()
            )
        ]
        return sum(pesqs) / len(pesqs) if pesqs else 0
