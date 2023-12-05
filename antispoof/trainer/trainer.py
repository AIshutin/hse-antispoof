import random
from pathlib import Path
from random import shuffle

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from antispoof.base import BaseTrainer
from antispoof.utils import inf_loop, MetricTracker
from ..logger.wandb import WanDBWriter
import json
import os


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
        print('TRAIN DATALODER', self.train_dataloader)
        logger.info(f"Model size: {calc_params(model)/1e6:.4f}M")
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.log_step =  config['trainer'].get('log_step', 10)
        self.loss_keys = ['loss']

        self.train_metrics = MetricTracker(
            *self.loss_keys,
            "grad_norm",
            *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            *self.loss_keys,
            *[m.name for m in self.metrics], writer=self.writer
        )
        self.cnt = 0
        self.scheduler_steps = config['trainer']['scheduler_steps']
    
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

            for part, dataloader in self.evaluation_dataloaders.items():
                if part == "test":
                    val_log = self._evaluation_epoch_test(epoch, part, dataloader)
                else:
                    val_log = self._evaluation_epoch(epoch, part, dataloader)
                    log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

            return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        assert(type(outputs) is dict)
        batch.update(outputs)
        batch.update(self.criterion(**batch))
        
        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            self.cnt += 1
            if self.cnt % self.scheduler_steps == 0:
                self.lr_scheduler.step()

        for key in self.loss_keys:
            metrics.update(key, batch[key].item())
        for met in self.metrics:
            if met.name in metrics.tracked_metrics:
                metrics.update(met.name, met(**batch))
        return batch
    
    def process_test_batch(self, batch):
        batch = self.move_batch_to_device(batch, self.device)
        outputs = self.model(spectrogram=batch['spectrogram'])
        assert(type(outputs) is dict)
        batch.update(outputs)
        return batch


    def _evaluation_epoch_test(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        return 
        with torch.no_grad():
            batches = []
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_test_batch(batch)
                batches.append(batch)
            
            self.writer.set_step(epoch * self.len_epoch, part)
            rows = {}
            for i, batch in enumerate(batches):
                audio_path = batch['audio_path']
                rows[str(i) + audio_path[0]] = {
                    "audio_hat": make_audio_item(batch['audio_hat'], self.writer, sample_rate=self.sr),
                    "audio": make_audio_item(batch['audio'], self.writer, sample_rate=self.sr),
                    "name": audio_path
                }
            self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))


    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
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
            if part != "test":
                self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch, is_test=part == "test")
            # self._log_audio(batch["audio"], batch['audio_hat'])

        # add histogram of model parameters to the tensorboard
        #for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins="auto")
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
            audio,
            target,
            target_hat,
            audio_path,
            examples_to_log=10,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return
        
        predictions = F.softmax(target_hat, dim=-1).argmax(dim=-1)
        tuples = list(zip(audio, target, predictions, audio_path))
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
