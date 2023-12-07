import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            sr,
            spec_processing,
            limit=None,
            max_audio_length=None,
            spec_segment_length=750
    ):

        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index, max_audio_length, limit)
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)
        self._index: List[dict] = index
        self.sr = sr
        self.spec_processing = spec_processing
        self.spec_segment_length = spec_segment_length

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_wave, audio_spec = self.load_audio(data_dict["audio"])
        if audio_spec.shape[-1] < self.spec_segment_length:
            audio_spec = torch.nn.functional.pad(audio_spec, (0, self.spec_segment_length - audio_spec.shape[-1]))
        else:
            n = random.randint(0, audio_spec.shape[-1] - self.spec_segment_length)
            audio_spec = audio_spec[:, :, n:n + self.spec_segment_length]

        out_dict = {
            "audio": audio_wave,
            "spec": audio_spec,
            "duration": audio_wave.size(1) / self.sr,
            "audio_path": data_dict["audio"],
            "target": data_dict['target']
        }
        return out_dict

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_length"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        spec = self.spec_processing(audio_tensor)
        return audio_tensor, spec

    @staticmethod
    def _filter_records_from_dataset(
            index: list, max_audio_length, limit
    ) -> list:
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = np.array([el["audio_length"] for el in index]) >= max_audio_length
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
            assert "audio_length" in entry, (
                "Each dataset item should include field 'audio_len'"
                " - duration of audio (in seconds)."
            )
            assert "audio" in entry, (
                "Each dataset item should include field 'audio'"
                " - path to audio file."
            )
            assert "target" in entry, (
                "Each dataset item should include field 'target'"
                " - target. 1 is bonafide, 0 is spoof"
            )
            