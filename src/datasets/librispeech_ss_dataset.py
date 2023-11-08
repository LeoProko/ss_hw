import os
from grob import glob

import torch


class LibrispeechSSDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, limit: int | None = None):
        self.ref_train = sorted(glob(os.path.join(data_dir, '*-ref.wav')))
        self.mix_train = sorted(glob(os.path.join(data_dir, '*-mixed.wav')))
        self.target_train = sorted(glob(os.path.join(data_dir, '*-target.wav')))

        assert len(ref_train) == len(mix_train) == len(target_train)

        if limit is None:
            limit = len(ref_train)

        self.ref_train = self.ref_train[:limit]
        self.mix_train = self.mix_train[:limit]
        self.target_train = self.target_train[:limit]


    def __getitem__(self, idx):
        return {
            "ref": self.ref_train[idx],
            "mix": self.mix_train[idx],
            "target": self.target_train[idx],
        }

    def __len__(self):
        return len(self.ref_train)