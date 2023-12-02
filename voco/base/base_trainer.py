from abc import abstractmethod

import torch
from numpy import inf
import wandb

from voco.base import BaseModel
from voco.logger import get_visualizer
import numpy as np
import logging
from pathlib import Path


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, generator, discriminator, g_criterion, d_criterion, metrics, 
                 g_optimizer, d_optimizer, config, device, logger):
        self.device = device
        self.logger = logger

        self.generator = generator
        self.discriminator = discriminator
        self.g_criterion = g_criterion
        self.d_criterion = d_criterion
        self.metrics = metrics
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
  
        # for interrupt saving
        self._last_epoch = 0

        self.config = config
        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = Path(cfg_trainer["save_dir"])

        # setup visualization writer instance
        self.writer = get_visualizer(
            config, self.logger, cfg_trainer["visualize"]
        )

        if config.get("resume") is not None:
            self._resume_checkpoint(config["resume"])

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError()

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not,
                    # according to specified metric(mnt_metric)
                    if self.mnt_mode == "min":
                        improved = log[self.mnt_metric] <= self.mnt_best
                    elif self.mnt_mode == "max":
                        improved = log[self.mnt_metric] >= self.mnt_best
                    else:
                        improved = False
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        g_arch = type(self.generator).__name__
        d_arch = type(self.discriminator).__name__
        state = {
            "g_arch": g_arch,
            "d_arch": d_arch,
            "epoch": epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "g_optimizer": self.g_optimizer.state_dict(),
            "d_optimizer": self.d_optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        if hasattr(self, 'd_lr_scheduler'):
            state['d_lr_scheduler'] = self.d_lr_scheduler.state_dict()
        if hasattr(self, 'g_lr_scheduler'):
            state['g_lr_scheduler'] = self.g_lr_scheduler.state_dict()
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
                checkpoint["config"]["g_optimizer"] != self.config["g_optimizer"] or
                checkpoint["config"]["g_lr_scheduler"] != self.config["g_lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            self.g_optimizer.load_state_dict(checkpoint["g_optimizer"])
            if 'g_lr_scheduler' in checkpoint:
                    self.g_lr_scheduler.load_state_dict(checkpoint['g_lr_scheduler'])

                # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
                checkpoint["config"]["d_optimizer"] != self.config["d_optimizer"] or
                checkpoint["config"]["d_lr_scheduler"] != self.config["d_lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            self.d_optimizer.load_state_dict(checkpoint["d_optimizer"])
            if 'd_lr_scheduler' in checkpoint:
                self.d_lr_scheduler.load_state_dict(checkpoint['d_lr_scheduler'])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
