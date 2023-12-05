import json
import logging
import os
import shutil
from curses.ascii import isascii
from pathlib import Path

import torchaudio
from antispoof.base.base_dataset import BaseDataset
from antispoof.utils import ROOT_PATH
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

logger = logging.getLogger(__name__)

class LADataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "LA"
        self._data_dir = data_dir
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)


    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index


    def _create_index(self, part):
        index = []

        suffix = "trl"
        if part == "train":
            suffix = "trn"
        protocol_path = self._data_dir / f'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{part}.{suffix}.txt'
        audio_path = self._data_dir / f'ASVspoof2019_LA_{part}/flac'

        with open(protocol_path) as file:
            for line in file:
                # example: LA_0069 LA_D_1047731 - - bonafide
                ref, audio, _, _, cls = line.strip().split()
                index.append({
                    "audio": str(audio_path / (audio + '.flac')),
                    "target": 0 if 'bonafide' in cls else 1
                })
                t_info = torchaudio.info(str(index[-1]['audio']))
                audio_length = t_info.num_frames / t_info.sample_rate
                index[-1]['audio_length'] = audio_length
        return index
