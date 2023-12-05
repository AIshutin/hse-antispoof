from torch import nn
from torch.nn import Sequential

from antispoof.base import BaseModel


class PlaceholderModel(BaseModel):
    def __init__(self, **batch):
        super().__init__()
        self.net = Sequential(
            nn.Linear(in_features=1, out_features=10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

    def forward(self, spectrogram, **batch):
        spec = spectrogram.mean(dim=(-1, -2)).reshape(-1, 1)
        return {
            "target_hat": self.net(spec)
        }