import logging
from typing import List, Iterable
import torch
from torch.nn import functional as F
import numpy as np

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
    audio = [el['audio'].squeeze(0) for el in batch]
    spec = [el['spec'].squeeze(0) for el in batch]
    audio_paths = [el["audio_path"] for el in batch]
    target = torch.tensor([el['target'] for el in batch])
    audio = pad_1D_tensor(audio)
    spec = pad_mels(spec)

    return {
        "audio": audio,
        "spectrogram": spec,
        "target": target,
        "audio_path": audio_paths,
    }


def get_collate_fn():
    return collate_fn_tensor