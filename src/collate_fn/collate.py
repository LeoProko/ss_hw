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

    refs = [item["ref"] for item in dataset_items]

    mixs = [item["mix"] for item in dataset_items]
    targets = [item["target"] for item in dataset_items]
    lengths = [item["target"].size(-1) for item in dataset_items]
    speaker_ids = [item["speaker_id"] for item in dataset_items]

    max_len = max(lengths)

    mixs = pad_sequence(
        [pad(mix[:max_len], (0, max_len - mix.size(-1))) for mix in mixs],
        batch_first=True,
    )
    targets = pad_sequence(
        [pad(target[:max_len], (0, max_len - target.size(-1))) for target in targets],
        batch_first=True,
    )
    refs = pad_sequence(
        [
            pad(ref, (0, max([ref.size(-1) for ref in refs]) - ref.size(-1)))
            for ref in refs
        ],
        batch_first=True,
    )

    return {
        "mix": mixs,
        "ref": refs,
        "target": targets,
        "length": torch.tensor(lengths),
        "speaker_id": torch.tensor(speaker_ids),
    }
