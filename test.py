import warnings

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import yaml
import json
from datetime import datetime
from antispoof.trainer.trainer import calc_params
import wandb
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@hydra.main(version_base=None, config_path="antispoof/configs/", config_name="lcnn_cqt_test")
def main(config: DictConfig):
    SEED = config.get('seed', 123)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    config2 = yaml.safe_load(OmegaConf.to_yaml(config))
    run_id = datetime.now().strftime(r"%m%d_%H%M%S")

    logger = instantiate(config.logger, main_config=json.dumps(config2), run_id=run_id)
    device = instantiate(config.device)
    model = instantiate(config.arch).to(device)
    checkpoint = torch.load(config.checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    logger.info(f"Model params: {calc_params(model)/1e6:.4f}M")
    logger.info(f"Model head params: {calc_params(model.head)/1e6:.4f}M")
    test_dataloader = instantiate(config.data)['test']
    thr = config.thr
    logger.info(model)
    model.eval()
    with torch.no_grad():
        rows = []

        for batch in test_dataloader:
            spectrogram = batch['spectrogram'].to(device)
            score = model(spectrogram)['score']
            rows.append({
                "audio": wandb.Audio(batch['audio'].detach().cpu().flatten().numpy(), sample_rate=config.trainer.sr),
                "score": score.item(),
                "is_bonafide": score.item() >= thr,
                "audio_path": batch['audio_path'][0]
            })

        df = pd.DataFrame(rows)
        wandb.init(project=config.trainer.wandb_project)
        wandb.log({
            "predictions": wandb.Table(dataframe=df)
        })


if __name__ == "__main__":
    main()