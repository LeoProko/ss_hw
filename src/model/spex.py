import torch

from src import base


class ResNetBlock(torch.nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Conv1d(in_dims, out_dims, 1),
            torch.nn.BatchNorm1d(out_dims),
            torch.nn.PReLU(),
            torch.nn.Conv1d(out_dims, out_dims, 1),
            torch.nn.BatchNorm1d(out_dims),
        )
        self.prelu = torch.nn.PReLU()
        self.maxpool = torch.nn.MaxPool1d(3)

    def forward(self, input):
        x = self.seq(input) + input
        x = self.prelu(x)
        x = self.maxpool(x)

        return x


class EncoderBlock(torch.nn.Module):
    def __init__(
        self,
        short_window_len,
        middle_window_len,
        long_window_len,
        decoder_dim,
    ):
        super().__init__()

        self.encoder_short = torch.nn.Sequential(
            torch.nn.Conv1d(1, decoder_dim, short_window_len, short_window_len // 2),
            torch.nn.ReLU(),
        )
        self.encoder_middle = torch.nn.Sequential(
            torch.nn.Conv1d(1, decoder_dim, middle_window_len, short_window_len // 2),
            torch.nn.ReLU(),
        )
        self.encoder_long = torch.nn.Sequential(
            torch.nn.Conv1d(1, decoder_dim, long_window_len, short_window_len // 2),
            torch.nn.ReLU(),
        )

    def forward(self, audio):
        x_1 = self.encoder_short(audio)
        x_2 = self.encoder_middle(audio)
        x_3 = self.encoder_long(audio)

        x_1 = x_1[:, :, : x_3.size(-1)]
        x_2 = x_2[:, :, : x_3.size(-1)]

        return x_1, x_2, x_3


class DecoderBlock(torch.nn.Module):
    def __init__(
        self,
        tcn_in_channels,
        short_window_len,
        middle_window_len,
        long_window_len,
        decoder_dim,
    ):
        super().__init__()

        self.m_1 = torch.nn.Sequential(
            torch.nn.Conv1d(tcn_in_channels, decoder_dim, 1),
            torch.nn.ReLU(),
        )
        self.m_2 = torch.nn.Sequential(
            torch.nn.Conv1d(tcn_in_channels, decoder_dim, 1),
            torch.nn.ReLU(),
        )
        self.m_3 = torch.nn.Sequential(
            torch.nn.Conv1d(tcn_in_channels, decoder_dim, 1),
            torch.nn.ReLU(),
        )

        self.decoder_short = torch.nn.ConvTranspose1d(
            decoder_dim, 1, kernel_size=short_window_len, stride=short_window_len // 2
        )
        self.decoder_middle = torch.nn.ConvTranspose1d(
            decoder_dim, 1, kernel_size=middle_window_len, stride=short_window_len // 2
        )
        self.decoder_long = torch.nn.ConvTranspose1d(
            decoder_dim, 1, kernel_size=long_window_len, stride=short_window_len // 2
        )

    def forward(self, y_1, y_2, y_3, y):
        m_1 = self.m_1(y)
        m_2 = self.m_2(y)
        m_3 = self.m_3(y)

        s_1 = self.decoder_short(m_1 * y_1)
        s_2 = self.decoder_middle(m_2 * y_2)
        s_3 = self.decoder_long(m_3 * y_3)

        return s_1, s_2, s_3


class TCNBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
    ):
        super().__init__()

        self.prelu = torch.nn.PReLU()
        self.cnn_1 = torch.nn.Conv1d(in_channels, out_channels, 1)
        self.norm_1 = torch.nn.LayerNorm(out_channels)
        self.de_cnn = torch.nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            groups=out_channels,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
        )
        self.norm_2 = torch.nn.LayerNorm(out_channels)
        self.cnn_2 = torch.nn.Conv1d(out_channels, in_channels, 1)

    @staticmethod
    def make_seq(depth, **kwargs):
        return torch.nn.Sequential(
            *[TCNBlock(**kwargs, dilation=2**i) for i in range(1, depth)]
        )

    def forward(self, input):
        x = self.cnn_1(input)
        x = self.prelu(x)
        x = self.norm_1(x.transpose(1, 2)).transpose(1, 2)
        x = self.de_cnn(x)
        x = self.prelu(x)
        x = self.norm_2(x.transpose(1, 2)).transpose(1, 2)
        x = self.cnn_2(x)

        return x + input


class SpeakexExtractorBlock(torch.nn.Module):
    def __init__(
        self,
        speakers_emb_dim,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
    ):
        super().__init__()

        assert dilation * (kernel_size - 1) % 2 == 0

        self.prelu = torch.nn.PReLU()
        self.cnn_1 = torch.nn.Conv1d(in_channels + speakers_emb_dim, out_channels, 1)
        self.norm_1 = torch.nn.LayerNorm(out_channels)
        self.de_cnn = torch.nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            groups=out_channels,
            padding=(dilation * (kernel_size - 1)) // 2,
            dilation=dilation,
        )
        self.norm_2 = torch.nn.LayerNorm(out_channels)
        self.cnn_2 = torch.nn.Conv1d(out_channels, in_channels, 1)

    def forward(self, mix, ref):
        ref = ref.unsqueeze(-1)
        ref = ref.repeat(1, 1, mix.size(-1))
        y = torch.cat([mix, ref], 1)
        y = self.cnn_1(y)
        y = self.norm_1(self.prelu(y).transpose(1, 2)).transpose(1, 2)
        y = self.de_cnn(y)
        y = self.norm_2(self.prelu(y).transpose(1, 2)).transpose(1, 2)
        y = self.cnn_2(y)
        y += mix
        return y


class SpEx(base.BaseModel):
    def __init__(
        self,
        short_window_len,
        middle_window_len,
        long_window_len,
        decoder_dim,
        res_blocks_num,
        tcn_depth,
        tcn_in_channels,
        tcn_out_channels,
        tcn_kernel_size,
        speakers_num,
        speakers_emb_dim,
        **batch,
    ):
        super().__init__(**batch)

        self.short_window_len = short_window_len

        self.encoder = EncoderBlock(
            short_window_len, middle_window_len, long_window_len, decoder_dim
        )
        self.decoder = DecoderBlock(
            tcn_in_channels,
            short_window_len,
            middle_window_len,
            long_window_len,
            decoder_dim,
        )

        self.y_norm = torch.nn.LayerNorm(3 * decoder_dim)
        self.y_cnn = torch.nn.Conv1d(3 * decoder_dim, tcn_in_channels, 1)

        self.speaker_extractor_1 = SpeakexExtractorBlock(
            speakers_emb_dim,
            tcn_in_channels,
            tcn_out_channels,
            tcn_kernel_size,
            1,
        )
        self.tcn_seq_1 = TCNBlock.make_seq(
            depth=tcn_depth,
            in_channels=tcn_in_channels,
            out_channels=tcn_out_channels,
            kernel_size=tcn_kernel_size,
        )
        self.speaker_extractor_2 = SpeakexExtractorBlock(
            speakers_emb_dim,
            tcn_in_channels,
            tcn_out_channels,
            tcn_kernel_size,
            1,
        )
        self.tcn_seq_2 = TCNBlock.make_seq(
            depth=tcn_depth,
            in_channels=tcn_in_channels,
            out_channels=tcn_out_channels,
            kernel_size=tcn_kernel_size,
        )
        self.speaker_extractor_3 = SpeakexExtractorBlock(
            speakers_emb_dim,
            tcn_in_channels,
            tcn_out_channels,
            tcn_kernel_size,
            1,
        )
        self.tcn_seq_3 = TCNBlock.make_seq(
            depth=tcn_depth,
            in_channels=tcn_in_channels,
            out_channels=tcn_out_channels,
            kernel_size=tcn_kernel_size,
        )
        self.speaker_extractor_4 = SpeakexExtractorBlock(
            speakers_emb_dim,
            tcn_in_channels,
            tcn_out_channels,
            tcn_kernel_size,
            1,
        )
        self.tcn_seq_4 = TCNBlock.make_seq(
            depth=tcn_depth,
            in_channels=tcn_in_channels,
            out_channels=tcn_out_channels,
            kernel_size=tcn_kernel_size,
        )

        self.speakers_num = speakers_num

        self.ref_norm = torch.nn.LayerNorm(3 * decoder_dim)
        self.encoded_ref_seq = torch.nn.Sequential(
            *[torch.nn.Conv1d(3 * decoder_dim, tcn_in_channels, 1)]
            + [
                ResNetBlock(tcn_in_channels, tcn_in_channels)
                for _ in range(res_blocks_num)
            ]
            + [torch.nn.Conv1d(tcn_in_channels, speakers_emb_dim, 1)]
        )

        self.pred_linear = torch.nn.Linear(speakers_emb_dim, speakers_num)

    def extract_speaker(self, y, x):
        y = self.speaker_extractor_1(y, x)
        y = self.tcn_seq_1(y)
        y = self.speaker_extractor_2(y, x)
        y = self.tcn_seq_2(y)
        y = self.speaker_extractor_3(y, x)
        y = self.tcn_seq_3(y)
        y = self.speaker_extractor_4(y, x)
        y = self.tcn_seq_4(y)

        return y

    def forward(self, mix, ref, length, *args, **kwargs):
        y_1, y_2, y_3 = self.encoder(mix)
        y_123 = torch.cat([y_1, y_2, y_3], 1)

        x = torch.cat([*self.encoder(ref)], 1)

        y = self.y_norm(y_123.transpose(1, 2)).transpose(1, 2)
        y = self.y_cnn(y)

        x = self.encoded_ref_seq(self.ref_norm(x.transpose(1, 2)).transpose(1, 2))

        # authors' trick
        den = (
            (length - self.short_window_len) // (self.short_window_len // 2) + 1
        ) // (3**3)
        x = x.sum(dim=-1) / den.view(-1, 1).float()

        y = self.extract_speaker(y, x)

        s_1, s_2, s_3 = self.decoder(y_1, y_2, y_3, y)

        return s_1, s_2, s_3, self.pred_linear(x)
