from torch import nn
from torch.nn import functional as F
import torch
import torchaudio


class MFM2x1(nn.Module):
    def __init__(self, channels, channel_dim=-3) -> None:
        super().__init__()
        assert(channels % 2 == 0)
        self.k = channels // 2
        self.channel_dim = channel_dim

    def forward(self, X):
        assert(X.shape[self.channel_dim] == self.k * 2)
        a = len(list(X.shape[self.channel_dim + 1:] if self.channel_dim != -1 else [])) + 1
        new_shape = list(X.shape[:self.channel_dim]) + [self.k, 2]
        final_shape = list(X.shape[:self.channel_dim]) + [self.k]
        if self.channel_dim != -1:
            new_shape += list(X.shape[self.channel_dim + 1:])
            final_shape += list(X.shape[self.channel_dim + 1:])
        X = X.reshape(new_shape)
        X = X.max(dim=-a).values
        X = X.reshape(final_shape)
        return X


def kaiman_trick(m):
    if not isinstance(m, nn.Linear) and not isinstance(m, nn.Conv2d):
        return
    nn.init.zeros_(m.bias)
    nn.init.kaiming_normal_(m.weight)

class LightCNN(nn.Module):
    def __init__(self, ifc_size, dropout_p=0.1, dropout_fc=0.7, sr=16000, banks=False, kaiman=False) -> None:
        super().__init__()
        if banks:
            self.prelinear = nn.Linear(257, 60)
            torch.nn.init.zeros_(self.prelinear.bias)
            new_weights = torchaudio.functional.linear_fbanks(
                n_freqs=257,
                n_filter=60,
                f_max=sr / 2,
                f_min=0,
                sample_rate=sr
            ).T
            assert(new_weights.shape == self.prelinear.weight.shape)
            with torch.no_grad():
                self.prelinear.weight = nn.Parameter(new_weights)

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            MFM2x1(64)
        )

        self.pool = nn.MaxPool2d(2, stride=2)

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1),
            MFM2x1(64),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, kernel_size=3, padding=1),
            MFM2x1(96)
        )

        self.block3 = nn.Sequential(
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, kernel_size=1),
            MFM2x1(96),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=3, padding=1),
            MFM2x1(128),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            MFM2x1(128),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            MFM2x1(64),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=1),
            MFM2x1(64),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            MFM2x1(64),
            nn.Dropout(dropout_p)
        )

        self.head = nn.Sequential(
            nn.Linear(ifc_size, 160),
            MFM2x1(160, -1),
            nn.BatchNorm1d(80),
            nn.Dropout(dropout_fc),
            nn.Linear(80, 2)
        )

        if kaiman:
            self.init_weights()
    
    def init_weights(self):
        print('KAIMAN TRICK')
        self.block1.apply(kaiman_trick)
        self.block2.apply(kaiman_trick)
        self.block3.apply(kaiman_trick)
        self.block4.apply(kaiman_trick)
        self.head.apply(kaiman_trick)
    
    def forward(self, spectrogram, **batch):
        X = spectrogram
        if hasattr(self, 'prelinear'):
            X = self.prelinear(spectrogram.transpose(-1, -2)).transpose(-1, -2)
        X = X.unsqueeze(1)
        
        X = self.block1(X)
        X = self.pool(X)
        X = self.block2(X)
        X = self.pool(X)
        X = self.block3(X)
        X = self.pool(X)
        X = self.block4(X)
        X = self.pool(X)
        X = X.reshape(X.shape[0], -1)
        X = self.head(X)

        return {
            "target_hat": X
        }


if __name__ == "__main__":    
    X = torch.randn(16, 10, 3, 6)
    net = MFM2x1(10, channel_dim=1)
    print(net(X).shape)
    net = MFM2x1(10, channel_dim=-3)
    print(net(X).shape)
    net = MFM2x1(6, channel_dim=3)
    print(net(X).shape)
    net = MFM2x1(6, channel_dim=-1)
    print(net(X).shape)