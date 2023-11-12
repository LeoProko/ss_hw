import torch
from torch.nn.utils import clip_grad_norm_
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SISDR
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ
from tqdm.auto import tqdm
import pyloudnorm as pyln

from src.base import BaseTrainer
from src.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        config,
        device,
        dataloaders,
        lr_scheduler=None,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.ce_loss = torch.nn.CrossEntropyLoss()
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "snr_loss", "ce_loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "snr_loss", *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["mix", "ref", "target", "speaker_id", "length"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    @torch.no_grad()
    def normalize_audio(self, audios: torch.Tensor):
        meter = pyln.Meter(self.config["preprocessing"]["sr"])
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

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["snr_loss"].item() + batch["ce_loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_predictions(is_train=True, **batch)
                self._log_scalars(self.train_metrics)
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    @staticmethod
    def si_sdr(est, target):
        alpha = (target * est).sum(dim=-1) / (target.norm(dim=-1) ** 2 + 1e-8)
        return 20 * torch.log10(
            (alpha * target).norm(dim=-1) / ((alpha * target - est).norm(dim=-1) + 1e-8)
            + 1e-8
        )

    def compute_loss(self, batch, is_train):
        short_logits, middle_logits, long_logits, speaker_logits = self.model(**batch)

        batch["pred"] = self.normalize_audio(short_logits)

        target = batch["target"]

        max_size = (
            torch.tensor(
                [
                    short_logits.size(-1),
                    middle_logits.size(-1),
                    long_logits.size(-1),
                    target.size(-1),
                ]
            )
            .min()
            .item()
        )

        short_logits = short_logits[:, :, :max_size]
        middle_logits = middle_logits[:, :, :max_size]
        long_logits = long_logits[:, :, :max_size]
        target = target[:, :, :max_size]

        snr1 = self.si_sdr(short_logits, target)
        snr2 = self.si_sdr(middle_logits, target)
        snr3 = self.si_sdr(long_logits, target)
        snr_loss = (
            -0.8 * torch.sum(snr1) - 0.1 * torch.sum(snr2) - 0.1 * torch.sum(snr3)
        ) / batch["mix"].size(0)

        if is_train:
            ce_loss = self.ce_loss(speaker_logits, batch["speaker_id"])
            batch["ce_loss"] = 10 * ce_loss

        batch["snr_loss"] = snr_loss

        return batch

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()

        batch = self.compute_loss(batch, is_train)

        if is_train:
            (batch["snr_loss"] + batch["ce_loss"]).backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            metrics.update("ce_loss", batch["ce_loss"].item())
            metrics.update("total_loss", batch["snr_loss"].item() + batch["ce_loss"].item())

        metrics.update("snr_loss", batch["snr_loss"].item())

        for met in self.metrics:
            metrics.update(met.name, met(**batch))

        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(is_train=False, **batch)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
        self,
        is_train,
        target,
        pred,
        mix,
        examples_to_log=10,
        *args,
        **kwargs,
    ):
        if self.writer is None:
            return

        step = "train" if is_train else "val"

        for i in range(target.size(0)):
            self.writer.add_audio(
                f"{step}-{i}-mix",
                mix[i, 0],
                sample_rate=self.config["preprocessing"]["sr"],
            )
            self.writer.add_audio(
                f"{step}-{i}-target",
                target[i, 0],
                sample_rate=self.config["preprocessing"]["sr"],
            )
            self.writer.add_audio(
                f"{step}-{i}-pred",
                pred[i, 0],
                sample_rate=self.config["preprocessing"]["sr"],
            )

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
