import logging
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    specs = [item["spectrogram"].squeeze() for item in dataset_items]
    max_spec_len = max([spec.size(-1) for spec in specs])
    specs = pad_sequence(
        [pad(spec, (0, max_spec_len - spec.size(-1))) for spec in specs],
        batch_first=True,
    )

    return {
        "audio": pad_sequence(
            [item["audio"].squeeze() for item in dataset_items], batch_first=True
        ),
        "spectrogram": specs,
        "spectrogram_length": torch.tensor(
            [item["spectrogram"].squeeze().size(-1) for item in dataset_items],
        ),
        "duration": [item["duration"] for item in dataset_items],
        "text": [item["text"] for item in dataset_items],
        "text_encoded": pad_sequence(
            [item["text_encoded"].squeeze() for item in dataset_items], batch_first=True
        ),
        "text_encoded_length": torch.tensor(
            [len(item["text_encoded"].squeeze()) for item in dataset_items],
        ),
        "audio_path": [item["audio_path"] for item in dataset_items],
    }
