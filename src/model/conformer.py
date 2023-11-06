import torch
from torch import nn

from src.base import BaseModel


class ConvolutionModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        assert kernel_size % 2 == 1

        self.norm = nn.LayerNorm(input_dim)
        self.layers = nn.Sequential(
            nn.Conv1d(
                input_dim,
                2 * num_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GLU(1),
            nn.Conv1d(
                num_channels,
                num_channels,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=num_channels,
            ),
            nn.BatchNorm1d(num_channels),
            nn.SiLU(),
            nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, input):
        x = self.norm(input)
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = x.transpose(1, 2)
        return x + input


class FeedForwardModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, input):
        return self.layers(input) * 0.5 + input


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_attention_heads: int,
        dropout: float,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(input_dim)
        self.attention = nn.MultiheadAttention(input_dim, num_attention_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, key_padding_mask):
        x = self.norm(input)
        x, _ = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.dropout(x)

        return x + input


class ConformerBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.ffn1 = FeedForwardModule(input_dim, ffn_dim, dropout)

        self.attention = MultiHeadSelfAttentionModule(
            input_dim, num_attention_heads, dropout
        )

        self.conv = ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.ffn2 = FeedForwardModule(input_dim, ffn_dim, dropout)

        self.norm = nn.LayerNorm(input_dim)

    def forward(self, input, key_padding_mask):
        x = self.ffn1(input)
        x = self.attention(x, key_padding_mask)
        x = self.conv(x)
        x = self.ffn2(x)
        x = self.norm(x)

        return x


class Conformer(BaseModel):
    def __init__(
        self,
        n_class: int,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
        **batch,
    ):
        super().__init__(input_dim, n_class, **batch)

        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    kernel_size,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.logits_layer = nn.Linear(in_features=input_dim, out_features=n_class)

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
