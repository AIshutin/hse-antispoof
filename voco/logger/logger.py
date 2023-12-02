import logging
import logging.config
from pathlib import Path

from voco.utils import read_json, write_yaml, ROOT_PATH
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import json

log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}


def setup_logging(
        save_dir, log_config=None, default_level=logging.INFO
):
    """
    Setup logging configuration
    """
    if log_config is None:
        log_config = str(ROOT_PATH / "voco" / "logger" / "logger_config.json")
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])

        logging.config.dictConfig(config)
    else:
        print(
            "Warning: logging configuration file is not found in {}.".format(log_config)
        )
        logging.basicConfig(level=default_level)


def get_logger(name, experiment_name, save_dir, run_id, main_config=None, verbosity=2):
    msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(
        verbosity, log_levels.keys()
    )
    assert verbosity in log_levels, msg_verbosity

    logger = logging.getLogger(name)
    logger.setLevel(log_levels[verbosity])

    save_dir = Path(save_dir) / run_id 
    log_dir = Path(save_dir) / "log"

    # make directory for saving checkpoints and log.
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # save updated config file to the checkpoint dir
    if main_config is not None:
        main_config = json.loads(main_config)
        with open(save_dir / "config.json", "w") as f:
            print(OmegaConf.to_yaml(main_config), file=f)

    # configure logging module
    setup_logging(log_dir)
    return logger