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

    refs = [item["ref"].squeeze() for item in dataset_items]
    mixs = [item["ref"].squeeze() for item in dataset_items]
    targets = [item["ref"].squeeze() for item in dataset_items]
    lengths = [mix.size(-1) for mix in mixs]
    max_spec_len = max(lengths)

    mixs = pad_sequence(
        [pad(mix, (0, max_spec_len - mix.size(-1))) for mix in mixs],
        batch_first=True,
    )
    refs = pad_sequence(
        [pad(ref, (0, max_spec_len - ref.size(-1))) for ref in refs],
        batch_first=True,
    )
    targets = pad_sequence(
        [pad(target, (0, max_spec_len - target.size(-1))) for target in targets],
        batch_first=True,
    )

    return {
        "mix": mixs,
        "ref": refs,
        "target": targets,
        "length": torch.tensor(lengths),
    }
