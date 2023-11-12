import os
from glob import glob

import torch

from src.base.base_dataset import BaseDataset


class LibrispeechSSDataset(BaseDataset):
    def __init__(self, data_dir: str, limit=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ref = sorted(glob(os.path.join(data_dir, "*-ref.wav")))
        self.mix = sorted(glob(os.path.join(data_dir, "*-mixed.wav")))
        self.target = sorted(glob(os.path.join(data_dir, "*-target.wav")))

        assert len(self.ref) == len(self.mix) == len(self.target)

        if limit is None:
            limit = len(self.ref)

        speakers = set(
            self.ref[i].split("/")[-1].split("_")[0] for i in range(len(self.ref))
        )
        self.speakers_map = dict((speaker, i) for i, speaker in enumerate(speakers))

        self.ref = self.ref[:limit]
        self.mix = self.mix[:limit]
        self.target = self.target[:limit]

    def __getitem__(self, idx):
        return {
            "ref": self.load_audio(self.ref[idx]),
            "mix": self.load_audio(self.mix[idx]),
            "target": self.load_audio(self.target[idx]),
            "speaker_id": self.speakers_map[self.ref[idx].split("/")[-1].split("_")[0]],
        }

    def __len__(self):
        return len(self.ref)
