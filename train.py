import warnings

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import yaml
import json
from datetime import datetime


warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


@hydra.main(version_base=None, config_path="voco/configs/", config_name="hifigan")
def main(config: DictConfig):
    config2 = yaml.safe_load(OmegaConf.to_yaml(config))
    run_id = datetime.now().strftime(r"%m%d_%H%M%S")

    logger = instantiate(config.logger, main_config=json.dumps(config2), run_id=run_id)
    device = instantiate(config.device)
    generator = instantiate(config.generator).to(device)
    discriminator = instantiate(config.discriminator).to(device)
    logger.info(generator)
    logger.info(discriminator)
    g_criterion = instantiate(config.g_loss).to(device)
    d_criterion = instantiate(config.d_loss).to(device)
    metrics = [
        instantiate(el) for el in config.metrics
    ]
    g_trainable_params = filter(lambda p: p.requires_grad, generator.parameters())
    d_trainable_params = filter(lambda p: p.requires_grad, discriminator.parameters())
    g_optimizer = instantiate(config.g_optimizer, params=g_trainable_params)
    d_optimizer = instantiate(config.d_optimizer, params=d_trainable_params)
    g_lr_scheduler = instantiate(config.g_lr_scheduler, optimizer=g_optimizer)
    d_lr_scheduler = instantiate(config.d_lr_scheduler, optimizer=d_optimizer)
    dataloaders = instantiate(config.data)

    trainer = instantiate(
        config.trainer,
        generator=generator,
        discriminator=discriminator,
        g_criterion=g_criterion,
        d_criterion=d_criterion,
        g_lr_scheduler=g_lr_scheduler,
        d_lr_scheduler=d_lr_scheduler,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        metrics=metrics,
        device=device,
        dataloaders=dataloaders,
        logger=logger,
        main_config=json.dumps(config2),
        run_id=run_id
    )
    trainer.train()


if __name__ == "__main__":
    main()