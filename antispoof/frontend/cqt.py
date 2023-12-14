import torch
import torchaudio
import librosa
import numpy as np


class CQT:
    def __init__(self, sr, hop_length, n_dim=84, n_dim_per_octave=12, **kwargs) -> None:
        self.sr = sr
        self.hop_length = hop_length
        self.n_bins = n_dim
        self.bins_per_octave = n_dim_per_octave
    
    def __call__(self, X):
        X = np.abs(librosa.cqt(X.numpy(), sr=self.sr, hop_length=self.hop_length, n_bins=self.n_bins,
                               bins_per_octave=self.bins_per_octave))
        X = torch.from_numpy(X)
        return torch.log(X + 1)