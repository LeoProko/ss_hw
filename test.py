import argparse
import json
import os
from pathlib import Path

import torch
from tqdm.auto import tqdm
import pyloudnorm as pyln
import hydra

import src.model as module_model
from src.trainer import Trainer
from src.utils import ROOT_PATH
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser
from src.metric import sisdr, pesq

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


@torch.no_grad()
def normalize_audio(sr, audios: torch.Tensor):
    meter = pyln.Meter(sr)
    audios = [audio.squeeze().detach().cpu().numpy() for audio in audios]

    return torch.stack(
        [
            torch.from_numpy(
                pyln.normalize.loudness(
                    audio,
                    meter.integrated_loudness(audio),
                    -20.0,
                )
            )
            for audio in audios
        ]
    ).unsqueeze(1)


@hydra.main()
def main(config):
    config = ConfigParser(config)
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config["resume"]))
    checkpoint = torch.load(config["resume"], map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    sisdr_metric = sisdr.SISDRMetric(device)
    pesq_metric = pesq.PESQMetric(config["preprocessing"]["sr"], device)

    sisdr_avg = 0
    pesq_avg = 0

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)

            short_logits, _, _, _ = model(**batch)

            batch["pred"] = normalize_audio(config["preprocessing"]["sr"], short_logits)

            sisdr_avg += sisdr_metric(**batch)
            pesq_avg += pesq_metric(**batch)

    sisdr_avg /= len(dataloaders["test"])
    pesq_avg /= len(dataloaders["test"])

    with Path("metrics_output.json").open("w") as fout:
        fout.write(f"sisdr: {sisdr_avg}\npesq: {pesq_avg}\n")


if __name__ == "__main__":
    main()
