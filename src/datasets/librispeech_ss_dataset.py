import os
from glob import glob

import torch

from src.base.base_dataset import BaseDataset


class LibrispeechSSDataset(BaseDataset):
    def __init__(self, data_dir: str, limit: int | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ref_train = sorted(glob(os.path.join(data_dir, '*-ref.wav')))
        self.mix_train = sorted(glob(os.path.join(data_dir, '*-mixed.wav')))
        self.target_train = sorted(glob(os.path.join(data_dir, '*-target.wav')))

        assert len(self.ref_train) == len(self.mix_train) == len(self.target_train)

        if limit is None:
            limit = len(ref_train)

        self.ref_train = self.ref_train[:limit]
        self.mix_train = self.mix_train[:limit]
        self.target_train = self.target_train[:limit]


    def get_spectorgram(self, audio_path: str):
        audio_wave = self.load_audio(audio_path)
        audio_wave, audio_spec = self.process_wave(audio_wave)
        return audio_spec

    def __getitem__(self, idx):
        return {
            "ref": self.get_spectorgram(self.ref_train[idx]),
            "mix": self.get_spectorgram(self.mix_train[idx]),
            "target": self.get_spectorgram(self.target_train[idx]),
        }

    def __len__(self):
        return len(self.ref_train)