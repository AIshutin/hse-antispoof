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
    print('Kaiman trick')
    if not isinstance(m, nn.Linear) and not isinstance(m, nn.Conv2d):
        return
    
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)
    nn.init.kaiming_normal_(m.weight)


class AMSoftmax(nn.Module):
    def __init__(self, m=0.2) -> None:
        super().__init__()
        self.m = m
        self.w = nn.Linear(2, 2, bias=False)
    
    def forward(self, X, fast=False):
        X_norm = torch.norm(X, dim=-1)
        X = X / X_norm.unsqueeze(-1)
        cos_theta = X @ (self.w.weight / torch.norm(self.w.weight, dim=-1, keepdim=True)).T

        assert((cos_theta < 1.01).all())
        assert((cos_theta > -1.01).all())


        theta = torch.acos(torch.clamp(cos_theta, min=-0.99999, max=0.99999))
        exp_wo_margins = torch.exp(cos_theta) # X * 
        exp_with_margins = torch.exp(torch.cos(theta * self.m)) # X * 
        p1 = exp_with_margins[:, 1] / (exp_wo_margins[:, 0] + exp_with_margins[:, 1])
        p0 = exp_with_margins[:, 0] / (exp_wo_margins[:, 1] + exp_with_margins[:, 0])
        score = exp_wo_margins[:, 1] / (exp_wo_margins.sum(dim=-1))

        out = { 
            "p1": p1,
            "p0": p0,
            "score": score
        }
        return out


class NormalSoftmax(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, X, fast):
        if not fast:
            probs = self.softmax(X)
            return {
                "logits": X,
                "p1": probs[:, 1],
                "p0": probs[:, 0],
                "score": probs[:, 1]
            }
        else:
            return {
                "logits": X,
                "p1": X[:, 1],
                "p0": X[:, 0],
                "score": (X[:, 1] > X[:, 0]).float()
            }


class LinearHead(nn.Module):
    def __init__(self, ifc, dropout, bias=True, kaiman=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ifc, 160),
            MFM2x1(160, -1),
            nn.BatchNorm1d(80),
            nn.Dropout(dropout),
            nn.Linear(80, 2, bias=bias)
        )
        if kaiman:
            self.net.apply(kaiman_trick)

    def forward(self, X):
        return self.net(X)


class LightCNNTrim(nn.Module):
    def __init__(self, backbone, head, softmax) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.softmax = softmax
    
    def forward(self, spectrogram, fast=False, **kwargs):
        X = self.backbone(spectrogram)
        X = self.head(X)
        X = self.softmax(X, fast)
        return X


class LightCNNBackbone(nn.Module):
    def __init__(self, dropout, kaiman) -> None:
        super().__init__()
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
            nn.Dropout(dropout)
        )

        if kaiman:
            self.block1.apply(kaiman_trick)
            self.block2.apply(kaiman_trick)
            self.block3.apply(kaiman_trick)
            self.block4.apply(kaiman_trick)

    
    def forward(self, spectrogram):
        X = spectrogram.unsqueeze(1)
        
        X = self.block1(X)
        X = self.pool(X)
        X = self.block2(X)
        X = self.pool(X)
        X = self.block3(X)
        X = self.pool(X)
        X = self.block4(X)
        X = self.pool(X)
        X = X.reshape(X.shape[0], -1)
        return X

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