import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F

from src.base import BaseModel


class TCNBlock(nn.Module):
    """
    Temporal convolutional network block,
        1x1Conv - PReLU - Norm - DConv - PReLU - Norm - SConv
    Input: 3D tensor witorch [N, C_in, L_in]
    Output: 3D tensor witorch [N, C_out, L_out]
    """

    def __init__(
        self,
        in_channels=256,
        conv_channels=512,
        kernel_size=3,
        dilation=1,
    ):
        super(TCNBlock, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.LayerNorm(conv_channels)
        dconv_pad = (
            (dilation * (kernel_size - 1)) // 2
        )
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.LayerNorm(conv_channels, elementwise_affine=True)
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.norm1(self.prelu1(y))
        y = self.dconv(y)
        y = self.norm2(self.prelu2(y))
        y = self.sconv(y)
        y += x
        return y


class TCNBlock_Spk(nn.Module):
    """
    Temporal convolutional network block,
        1x1Conv - PReLU - Norm - DConv - PReLU - Norm - SConv
        torche first tcn block takes additional speaker embedding as inputs
    Input: 3D tensor witorch [N, C_in, L_in]
    Input Speaker Embedding: 2D tensor witorch [N, D]
    Output: 3D tensor witorch [N, C_out, L_out]
    """

    def __init__(
        self,
        in_channels=256,
        spk_embed_dim=100,
        conv_channels=512,
        kernel_size=3,
        dilation=1,
    ):
        super(TCNBlock_Spk, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels + spk_embed_dim, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.LayerNorm(conv_channels, elementwise_affine=True)
        dconv_pad = (
            (dilation * (kernel_size - 1)) // 2
        )
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True,
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.LayerNorm(conv_channels, elementwise_affine=True)
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        self.dconv_pad = dconv_pad
        self.dilation = dilation

    def forward(self, x, aux):
        # Repeatedly concated speaker embedding aux to each frame of torche representation x
        T = x.shape[-1]
        aux = torch.unsqueeze(aux, -1)
        aux = aux.repeat(1, 1, T)
        y = torch.cat([x, aux], 1)
        y = self.conv1x1(y)
        y = self.norm1(self.prelu1(y))
        y = self.dconv(y)
        y = self.norm2(self.prelu2(y))
        y = self.sconv(y)
        y += x
        return y


class ResBlock(nn.Module):
    """
    Resnet block for speaker encoder to obtain speaker embedding
    ref to
        https://gitorchub.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
        and
        https://gitorchub.com/Jungjee/RawNet/blob/master/PyTorch/model_RawNet.py
    """

    def __init__(self, in_dims, out_dims):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_dims, out_dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_dims)
        self.batch_norm2 = nn.BatchNorm1d(out_dims)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.maxpool = nn.MaxPool1d(3)
        if in_dims != out_dims:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(
                in_dims, out_dims, kernel_size=1, bias=False
            )
        else:
            self.downsample = False

    def forward(self, x):
        y = self.conv1(x)
        y = self.batch_norm1(y)
        y = self.prelu1(y)
        y = self.conv2(y)
        y = self.batch_norm2(y)
        if self.downsample:
            y += self.conv_downsample(x)
        else:
            y += x
        y = self.prelu2(y)
        return self.maxpool(y)


class SpEx(BaseModel):
    def __init__(
        self,
        L1,
        L2,
        L3,
        N,
        B,
        O,
        P,
        Q,
        num_spks,
        spk_embed_dim=256,
        **batch,
    ):
        super().__init__(**batch)
        # n x S => n x N x T, S = 4s*8000 = 32000
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.encoder_1d_short = nn.Conv1d(1, N, L1, stride=L1 // 2, padding=0)
        self.encoder_1d_middle = nn.Conv1d(1, N, L2, stride=L1 // 2, padding=0)
        self.encoder_1d_long = nn.Conv1d(1, N, L3, stride=L1 // 2, padding=0)
        # before repeat blocks, always cLN
        self.ln = nn.LayerNorm(3 * N)
        # n x N x T => n x O x T
        self.proj = nn.Conv1d(3 * N, O, 1)
        self.conv_block_1 = TCNBlock_Spk(
            spk_embed_dim=spk_embed_dim,
            in_channels=O,
            conv_channels=P,
            kernel_size=Q,
            dilation=1,
        )
        self.conv_block_1_otorcher = self._build_stacks(
            num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, 
        )
        self.conv_block_2 = TCNBlock_Spk(
            spk_embed_dim=spk_embed_dim,
            in_channels=O,
            conv_channels=P,
            kernel_size=Q,
            dilation=1,
        )
        self.conv_block_2_otorcher = self._build_stacks(
            num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, 
        )
        self.conv_block_3 = TCNBlock_Spk(
            spk_embed_dim=spk_embed_dim,
            in_channels=O,
            conv_channels=P,
            kernel_size=Q,
            dilation=1,
        )
        self.conv_block_3_otorcher = self._build_stacks(
            num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, 
        )
        self.conv_block_4 = TCNBlock_Spk(
            spk_embed_dim=spk_embed_dim,
            in_channels=O,
            conv_channels=P,
            kernel_size=Q,
            dilation=1,
        )
        self.conv_block_4_otorcher = self._build_stacks(
            num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, 
        )
        # n x O x T => n x N x T
        self.mask1 = nn.Conv1d(O, N, 1)
        self.mask2 = nn.Conv1d(O, N, 1)
        self.mask3 = nn.Conv1d(O, N, 1)
        # using ConvTrans1D: n x N x T => n x 1 x To
        # To = (T - 1) * L // 2 + L
        self.decoder_1d_short = nn.ConvTranspose1d(
            N, 1, kernel_size=L1, stride=L1 // 2, bias=True
        )
        self.decoder_1d_middle = nn.ConvTranspose1d(
            N, 1, kernel_size=L2, stride=L1 // 2, bias=True
        )
        self.decoder_1d_long = nn.ConvTranspose1d(
            N, 1, kernel_size=L3, stride=L1 // 2, bias=True
        )
        self.num_spks = num_spks

        self.spk_encoder = nn.Sequential(
            nn.LayerNorm(3 * N),
            nn.Conv1d(3 * N, O, 1),
            ResBlock(O, O),
            ResBlock(O, P),
            ResBlock(P, P),
            nn.Conv1d(P, spk_embed_dim, 1),
        )

        self.pred_linear = nn.Linear(spk_embed_dim, num_spks)

    def _build_stacks(self, num_blocks, **block_kwargs):
        """
        Stack B numbers of TCN block, torche first TCN block takes torche speaker embedding
        """
        blocks = [
            TCNBlock(**block_kwargs, dilation=(2**b)) for b in range(1, num_blocks)
        ]
        return nn.Sequential(*blocks)



    def forward(self, spectrogram, **batch):
        # spectrogram: (B, input_dim, L)

        max_length = torch.max(batch["spectrogram_length"]).item()
        key_padding_mask = torch.arange(
            max_length, device=spectrogram.device
        ).expand(batch["spectrogram_length"].size(0), max_length) >= batch[
            "spectrogram_length"
        ].unsqueeze(
            1
        ).to(
            spectrogram.device
        )

        x = spectrogram.transpose(1, 2)
        for block in self.blocks:
            x = block(x, key_padding_mask.T)

        return {"logits": self.logits_layer(x)}
    # def forward(self, x, aux, aux_len):
    def forward(self, **batch):
        x = batch["mix"]
        aux = batch["target"]
        aux_len = batch["length"]

        # if x.dim() >= 3:
        #     raise RuntimeError(
        #         "accept 1/2D tensor as input, but got {:d}".format(
        #             x.dim()
        #         )
        #     )
        # # when inference, only one utt
        # if x.dim() == 1:
        #     x = torch.unsqueeze(x, 0)

        # n x 1 x S => n x N x T
        w1 = F.relu(self.encoder_1d_short(x))
        T = w1.shape[-1]
        xlen1 = x.shape[-1]
        xlen2 = (T - 1) * (self.L1 // 2) + self.L2
        xlen3 = (T - 1) * (self.L1 // 2) + self.L3
        w2 = F.relu(self.encoder_1d_middle(F.pad(x, (0, xlen2 - xlen1), "constant", 0)))
        w3 = F.relu(self.encoder_1d_long(F.pad(x, (0, xlen3 - xlen1), "constant", 0)))

        # n x 3N x T
        y = self.ln(torch.cat([w1, w2, w3], 1))
        # n x O x T
        y = self.proj(y)

        # speaker encoder (share params from speech encoder)
        aux_w1 = F.relu(self.encoder_1d_short(aux))
        aux_T_shape = aux_w1.shape[-1]
        aux_len1 = aux.shape[-1]
        aux_len2 = (aux_T_shape - 1) * (self.L1 // 2) + self.L2
        aux_len3 = (aux_T_shape - 1) * (self.L1 // 2) + self.L3
        aux_w2 = F.relu(
            self.encoder_1d_middle(F.pad(aux, (0, aux_len2 - aux_len1), "constant", 0))
        )
        aux_w3 = F.relu(
            self.encoder_1d_long(F.pad(aux, (0, aux_len3 - aux_len1), "constant", 0))
        )

        aux = self.spk_encoder(torch.cat([aux_w1, aux_w2, aux_w3], 1))
        aux_T = (aux_len - self.L1) // (self.L1 // 2) + 1
        aux_T = ((aux_T // 3) // 3) // 3
        aux = torch.sum(aux, -1) / aux_T.view(-1, 1).float()

        y = self.conv_block_1(y, aux)
        y = self.conv_block_1_otorcher(y)
        y = self.conv_block_2(y, aux)
        y = self.conv_block_2_otorcher(y)
        y = self.conv_block_3(y, aux)
        y = self.conv_block_3_otorcher(y)
        y = self.conv_block_4(y, aux)
        y = self.conv_block_4_otorcher(y)

        # n x N x T
        m1 = F.relu(self.mask1(y))
        m2 = F.relu(self.mask2(y))
        m3 = F.relu(self.mask3(y))
        S1 = w1 * m1
        S2 = w2 * m2
        S3 = w3 * m3

        return (
            self.decoder_1d_short(S1),
            self.decoder_1d_middle(S2)[:, :xlen1],
            self.decoder_1d_long(S3)[:, :xlen1],
            self.pred_linear(aux),
        )
