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
        lengths = [
            min(min(length[i], pred.size(-1)), target.size(-1))
            for i in range(len(length))
        ]

        pesqs = []

        for i in range(pred.size(0)):
            try:
                pesq = (
                    self.pesq(
                        pred[i, 0, : lengths[i]].to(self.device),
                        target[i, 0, : lengths[i]].to(self.device),
                    )
                    if torch.isfinite(pred[i, 0]).all()
                    else torch.tensor([-0.5]).to(self.device)
                )
            except:
                pesq = torch.tensor([-0.5]).to(self.device)

            pesqs.append(pesq)

        return (sum(pesqs) / len(pesqs)).squeeze().detach().cpu().item()
