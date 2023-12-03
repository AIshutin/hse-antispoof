from torch import nn
from torch.nn import functional as F
import torch
from typing import List
from torch.nn.utils import weight_norm, spectral_norm # since I use torch 2.0


class MetaConfiguration:
    '''
    This is a crouch for backward compability. You can remove this code if don't use clown pipeline
    '''
    weight_reinit: False


def get_activation():
    if MetaConfiguration:
        return nn.LeakyReLU(0.1)
    return nn.LeakyReLU()


MEAN = 0
STD = 0.01

class ResBlock(nn.Module):
    def __init__(self, channels, kernel, Dr) -> None:
        super().__init__()
        conv_blocks = []
        for i in range(len(Dr)):
            layers = []
            for j in range(len(Dr[i])):
                layers.append(get_activation())
                layers.append(weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel, 
                                                    dilation=Dr[i][j], padding='same')))
                if MetaConfiguration.weight_reinit:
                    layers[-1].weight.data.normal_(MEAN, STD)
            conv_blocks.append(nn.Sequential(*layers))
        
        self.blocks = nn.ModuleList(conv_blocks)

    def forward(self, X):
        for layer in self.blocks:
            X = X + layer(X)
        return X


class MRF(nn.Module): 
    def __init__(self, channels, kr, Dr):
        super().__init__()
        self.layers = nn.ModuleList(
            ResBlock(channels, kr[t], Dr[t]) for t in range(len(kr))
        )
    
    def forward(self, X):
        out = 0
        for el in self.layers:
            out = out + el(X)
        return out / len(self.layers) # adds stability and speed to training, but in theory is not needed


class GeneratorBlock(nn.Module):
    def __init__(self, kernel, channels, kr, Dr) -> None:
        super().__init__()
        stride = kernel // 2
        conv_t = weight_norm(nn.ConvTranspose1d(channels * 2, channels, kernel_size=kernel,
                                                     stride=stride, padding= (kernel - stride) // 2))
        # conv_t.weight.data.normal_(MEAN, STD)
        self.net = nn.Sequential(
            get_activation(),
            conv_t
        )
        
        self.mrf = MRF(channels, kr, Dr)
    
    def forward(self, X):
        X = self.net(X)
        return self.mrf(X)


class Generator(nn.Module):
    def __init__(self, in_channels: int, hu: int, ku: List[int], 
                 kr: List[int], Dr: List[List[int]], weight_reinit=False) -> None:
        super().__init__()
        self.prenet = weight_norm(nn.Conv1d(in_channels, hu, kernel_size=7, padding=3))
        MetaConfiguration.weight_reinit = weight_reinit
        if MetaConfiguration.weight_reinit:
            self.prenet.weight.data.normal_(MEAN, STD)
        self.net = nn.Sequential(
            *[GeneratorBlock(ku[l], hu // 2 ** (1 + l), kr, Dr) for l in range(len(ku))]
        )
        self.act = get_activation()
        self.postnet = weight_norm(nn.Conv1d(hu // 2 ** len(ku), 1, kernel_size=7, padding=3))
        if MetaConfiguration.weight_reinit:
            self.postnet.weight.data.normal_(MEAN, STD)
        self.tanh = nn.Tanh()
    
    def forward(self, spectrogram, **kwargs):
        X = self.prenet(spectrogram)
        X = self.net(X)
        X = self.act(X)
        X = self.postnet(X)
        X = self.tanh(X)
        X = X.reshape(X.shape[0], X.shape[-1])
        return {
            "audio_hat": X
        }


class MPD(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        layers = []
        prev_channels = 1
        for l in range(1, 5):
            channels = 2 ** (5 + l)
            layers.append(
                weight_norm(
                    nn.Conv2d(prev_channels, channels, kernel_size=(1, 5),
                              stride=(1, 3))
                )
            )
            layers.append(get_activation())
            prev_channels = channels
        layers.append(weight_norm(nn.Conv2d(prev_channels, 1024, kernel_size=(1, 5))))
        layers.append(get_activation())
        layers.append(weight_norm(nn.Conv2d(1024, 1, kernel_size=(1, 3))))
        self.net = nn.ModuleList(
            layers
        )
    
    def forward(self, X):
        B, L = X.shape
        if X.shape[-1] % self.period != 0:
            X = F.pad(X, (0, self.period - (X.shape[-1] % self.period)))
        X = X.reshape(B, 1, self.period, -1)
        
        features = []
        for layer in self.net:
            X = layer(X)
            if isinstance(layer, nn.Conv2d):
                features.append(X)
        assert(len(features) != 0)
        return features


class MSD(nn.Module):
    # Look here: https://arxiv.org/pdf/1910.06711.pdf
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        if scale == 1:
            weight_f = spectral_norm
        else:
            weight_f = weight_norm
        
        self.net = nn.ModuleList([
            weight_f(nn.Conv1d(1, 16, kernel_size=15, stride=1, padding='same')),
            get_activation(),
            weight_f(nn.Conv1d(16, 64, kernel_size=41, stride=4, groups=4, padding=20)),
            get_activation(),
            weight_f(nn.Conv1d(64, 256, kernel_size=41, stride=4, groups=16, padding=20)),
            get_activation(),
            weight_f(nn.Conv1d(256, 1024, kernel_size=41, stride=4, groups=64, padding=20)),
            get_activation(),
            weight_f(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, groups=256, padding=20)),
            get_activation(),
            weight_f(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding='same')),
            get_activation(),
            weight_f(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding='same'))
        ])

        if self.scale == 1:
            self.pooling = lambda x: x
        else:
            self.pooling = nn.AvgPool1d(scale, stride = scale // 2, padding=scale // 2)
    
    def forward(self, X):
        X = self.pooling(X)
        features = []
        for layer in self.net:
            X = layer(X)
            if isinstance(layer, nn.Conv1d):
                features.append(X)
        assert(len(features) != 0)
        return features


class Discriminator(nn.Module):
    def __init__(self, periods, scales) -> None:
        super().__init__()
        self.mpds = nn.ModuleList(
            MPD(period) for period in periods
        )
        self.msds = nn.ModuleList(
            MSD(scale) for scale in scales
        )
    
    def forward(self, audio, audio_hat, **kwargs):
        gt_features = []
        hat_features = []
        discrim_hat_p = []
        discrim_p = []
        total_discriminators = 0
        
        assert(audio.shape == audio_hat.shape)

        for net in self.mpds:
            total_discriminators += 1
            curr_features = net(audio)
            discrim_p.append(curr_features[-1].reshape(audio.shape[0], -1))
            gt_features += curr_features[:-1]

            curr_hat_features = net(audio_hat)
            discrim_hat_p.append(curr_hat_features[-1].reshape(audio.shape[0], -1))
            hat_features += curr_hat_features[:-1]

        return {
            "dfeats_hat": hat_features, 
            "dfeats": gt_features,
            "discrim_hat_p": discrim_hat_p,
            "discrim_p": discrim_p
        }
