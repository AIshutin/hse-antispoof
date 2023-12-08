import random
from pathlib import Path
from random import shuffle

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm
import numpy as np

from antispoof.base import BaseTrainer
from antispoof.utils import inf_loop, MetricTracker
from ..logger.wandb import WanDBWriter
from antispoof.utils import ROOT_PATH
import json
import os
from ..calculate_eer import compute_eer


def make_audio_item(wave, writer, sr):
    if not isinstance(writer, WanDBWriter):
        raise NotImplemented(f"{str(type(writer))};{writer} is not supported here")
    wave = wave.detach().cpu().flatten().numpy()
    return writer.wandb.Audio(wave, sample_rate=sr)


def calc_params(model):
    return sum(el.numel() for el in model.parameters() if el.requires_grad_)


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            device,
            dataloaders,
            main_config,
            logger,
            run_id,
            sr,
            lr_scheduler,
            len_epoch=None,
            skip_oom=True,
            **kwargs
    ):
        self.lr_scheduler = lr_scheduler
        self.sr = sr

        config = json.loads(main_config)
        config['trainer']['save_dir'] = os.path.join(config['trainer']['save_dir'], run_id)
        super().__init__(model=model, criterion=criterion, metrics=metrics,
                         optimizer=optimizer, device=device, 
                         logger=logger, config=config)
        self.skip_oom = skip_oom
        self.train_dataloader = dataloaders["train"]
        self.logger.info(f'Train dataloader size {len(self.train_dataloader)}')
        self.logger.info(f"Model size: {calc_params(model)/1e6:.4f}M")
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.log_step =  config['trainer'].get('log_step', 100)
        self.loss_keys = ['loss']

        self.train_metrics = MetricTracker(
            *self.loss_keys,
            "grad_norm",
            *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            *self.loss_keys,
            'eer', 'eer_thr',
            *[m.name for m in self.metrics], writer=self.writer
        )
    
    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["audio", "spectrogram", "target"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        with torch.autograd.set_detect_anomaly(True):
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
                self.train_metrics.update("grad_norm", self.get_grad_norm(self.model))
                if batch_idx % self.log_step == 0:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f}".format(
                            epoch, self._progress(batch_idx), batch["loss"].item()
                        )
                    )
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                    self._log_predictions(**batch)
                    # self._log_audio(batch["audio"], batch['audio_hat'])
                    self._log_scalars(self.train_metrics)
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                    last_train_metrics = self.train_metrics.result()
                    self.train_metrics.reset()
                if batch_idx >= self.len_epoch:
                    break
            log = last_train_metrics
            self.lr_scheduler.step()

            for part, dataloader in self.evaluation_dataloaders.items():
                val_log = self._evaluation_epoch(epoch, part, dataloader)
                log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

            return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch, fast=is_train)
        assert(type(outputs) is dict)
        batch.update(outputs)
        batch.update(self.criterion(**batch))
        
        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()

        for key in self.loss_keys:
            metrics.update(key, batch[key].item())
        for met in self.metrics:
            if met.name in metrics.tracked_metrics:
                metrics.update(met.name, met(**batch))
        return batch
        

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        scores = [[] for i in range(2)]
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
                for score, target in zip(batch["score"], batch['target']):
                    scores[target.item()].append(score.item())
            self.writer.set_step(epoch * self.len_epoch, part)
            if part != "test":
                self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch, is_test=part == "test")

        scores = [np.array(scores[0]), np.array(scores[1])]
        eer, eer_thr = compute_eer(scores[1], scores[0])
        out = self.evaluation_metrics.result()
        self.writer.add_scalar(f"{part}_eer", eer)
        self.writer.add_scalar(f"{part}_eer_thr", eer_thr)
        out['eer'] = eer
        out['eer_thr'] = eer_thr
        return out

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
            audio,
            target,
            score,
            audio_path,
            examples_to_log=10,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return
        tuples = list(zip(audio, target, score, audio_path))
        shuffle(tuples)
        rows = {}
        for audio, target, target_hat, audio_path in tuples[:examples_to_log]:
            rows[Path(audio_path).name] = {
                "audio": make_audio_item(audio, self.writer, sr=self.sr),
                "target": target.item(),
                "predicted": target_hat.item(),
                "name": Path(audio_path).name
            }
        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_audio(self, audio_batch, audio_hat_batch):
        idx = random.randint(0, audio_batch.shape[0] - 1)
        audio = audio_batch[idx].detach().cpu()
        audio_hat = audio_hat_batch[idx].detach().cpu()
        self.writer.add_audio("audio", audio, sample_rate=self.sr)
        self.writer.add_audio("audio_hat", audio_hat, sample_rate=self.sr)

    @torch.no_grad()
    def get_grad_norm(self, model, norm_type=2):
        parameters = model.parameters()
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
