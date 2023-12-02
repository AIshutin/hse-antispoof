from torch import nn
from torch.nn import Sequential

from voco.base import BaseModel


class PlaceholderModel(BaseModel):
    def __init__(self, n_class, **batch):
        super().__init__()
        self.net = Sequential(
            nn.Linear(in_features=1, out_features=1)
        )

    def forward(self, spectrogram, energy, alignment, pitch, **batch):
        return {
            "pitch_hat": pitch,
            "alignment_hat": alignment,
            "energy_hat": energy,
            "spectrogram_hat": spectrogram + self.net(spectrogram.unsqueeze(-1)).squeeze(-1)
        }