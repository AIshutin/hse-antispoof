import logging
from typing import List, Iterable
import torch
from torch.nn import functional as F
import numpy as np
from voco.audio.mel import config as mel_config

logger = logging.getLogger(__name__)


def pad_1D(inputs, PAD=0):
    assert(len(inputs[0].shape) == 1)

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_1D_tensor(inputs, PAD=0):
    assert(len(inputs[0].shape) == 1)

    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    assert(len(inputs[0].shape) == 2)

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_mels(inputs, pad_value=0):
    assert(len(inputs[0].shape) == 2)

    def pad(x, max_len):
        len = x.size(1)
        if len > max_len:
            raise ValueError("not max_len")

        x_padded = F.pad(x, (0, max_len - len,), value=pad_value)
        return x_padded

    max_len = max(x.size(1) for x in inputs)
    output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def collate_fn_tensor(batch: List[dict]):
    mel_targets = [el["spectrogram"].squeeze(0) for el in batch]
    audio_paths = [el["audio_path"] for el in batch]
    audio_durations = [el["duration"] for el in batch]
    audios = [el['audio'].squeeze(0) for el in batch]
    mel_length = torch.tensor([el.shape[-1] for el in mel_targets])

    mel_targets = pad_mels(mel_targets, pad_value=mel_config.pad_value)
    audios = torch.from_numpy(pad_1D(audios))


    return {
        "spectrogram": mel_targets,
        "spectrogram_length": mel_length,
        "duration": audio_durations,
        "audio_path": audio_paths,
        "audio": audios
    }


def get_collate_fn():
    return collate_fn_tensor