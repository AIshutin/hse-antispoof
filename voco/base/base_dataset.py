import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from voco.audio.mel import config as mel_config, audio2mel


logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            limit=None,
            max_audio_length=None,
            segment_length=None,
    ):

        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index, max_audio_length, limit)
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)
        self._index: List[dict] = index
        self.sr = mel_config.sr
        self.segment_length = segment_length

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_wave = self.load_audio(audio_path)
        if self.segment_length is not None:
            random_start = 0
            if audio_wave.shape[1] > self.segment_length:
                if len(self) < 100:
                    random_start = 0
                else:
                    random_start = torch.randint(size=(1, ), low=0, 
                                                 high=audio_wave.shape[1] - self.segment_length).item()
            audio_wave = audio_wave[:, random_start:random_start + self.segment_length]
        spec = audio2mel(audio_wave)

        out_dict = {
            "audio": audio_wave,
            "spectrogram": spec,
            "duration": audio_wave.size(1) / self.sr,
            "audio_path": audio_path,
        }
        return out_dict

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    @staticmethod
    def _filter_records_from_dataset(
            index: list, max_audio_length, limit
    ) -> list:
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = np.array([el["audio_len"] for el in index]) >= max_audio_length
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)

        records_to_filter = exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "audio_len" in entry, (
                "Each dataset item should include field 'audio_len'"
                " - duration of audio (in seconds)."
            )
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
