import torch
from torch import nn

from src.base import BaseModel


class SpEx(BaseModel):
    def __init__(
        self,
        n_feats: int,
        n_class: int,
        **batch,
    ):
        super().__init__(n_feats, n_class, **batch)

        self.logits_layer = nn.Linear(in_features=n_feats, out_features=n_class)

    def forward(self, spectrogram, **batch):
        # spectrogram: (B, input_dim, L)

        max_length = torch.max(batch["spectrogram_length"]).item()
        key_padding_mask = torch.arange(max_length, device=spectrogram.device).expand(
            batch["spectrogram_length"].size(0), max_length
        ) >= batch["spectrogram_length"].unsqueeze(1).to(spectrogram.device)

        x = spectrogram.transpose(1, 2)
        for block in self.blocks:
            x = block(x, key_padding_mask.T)

        return {"logits": self.logits_layer(x)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
