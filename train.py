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


@hydra.main(version_base=None, config_path="antispoof/configs/", config_name="unittest")
def main(config: DictConfig):
    config2 = yaml.safe_load(OmegaConf.to_yaml(config))
    run_id = datetime.now().strftime(r"%m%d_%H%M%S")

    logger = instantiate(config.logger, main_config=json.dumps(config2), run_id=run_id)
    device = instantiate(config.device)
    model = instantiate(config.arch).to(device)
    logger.info(model)
    criterion = instantiate(config.loss).to(device)
    metrics = [
        instantiate(el) for el in config.metrics
    ]
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
    dataloaders = instantiate(config.data)

    trainer = instantiate(
        config.trainer,
        model=model,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        optimizer=optimizer,
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