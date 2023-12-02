import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn
from voco.audio.mel import MelSpectrogram, config as mel_config


class MelLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()
        self.audio2mel = MelSpectrogram(mel_config)

    def forward(self, audio_hat, spectrogram, **kwargs):
        return self.l1(self.audio2mel(audio_hat), spectrogram)


class GANLoss_DG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l2 = nn.MSELoss()
    
    def forward(self, discrim_hat_p, discrim_p, **kwargs):
        loss = 0
        for hat_p, p in zip(discrim_hat_p, discrim_p):
            loss += self.l2(p, torch.ones_like(p))
            loss += self.l2(hat_p, torch.zeros_like(hat_p))
        return {
            "d_loss": loss
        }

class GANLoss_GD(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l2 = nn.MSELoss()
    
    def forward(self, discrim_hat_p, **kwargs):
        loss = 0
        for hat_p in discrim_hat_p:
            loss += self.l2(hat_p, torch.ones_like(hat_p))
        return loss


class FMLoss(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.l2 = nn.L1Loss()

    def forward(self, dfeats_hat, dfeats, **kwargs):
        total = 0
        for hat, gt in zip(dfeats_hat, dfeats):
            total += self.l2(hat, gt)
        return total


class HiFiGANLoss_G(nn.Module):
    def __init__(self, fm=0.0, gan=0.0, mel=0.0) -> None:
        super().__init__()
        self.fm = fm
        self.gan = gan
        self.mel = mel
        if fm != 0:
            self.fm_loss = FMLoss()
        else:
            self.fm_loss = lambda **kwargs: torch.tensor(0)
        
        if gan != 0:
            self.gan_loss = GANLoss_GD()
        else:
            self.gan_loss = lambda **kwargs: torch.tensor(0)
        
        if mel != 0:
            self.mel_loss = MelLoss()
        else:
            self.mel_loss = lambda **kwargs: torch.tensor(0)
    
    def forward(self, **kwargs):
        fm_loss = self.fm_loss(**kwargs)
        gan_loss = self.gan_loss(**kwargs)
        mel_loss = self.mel_loss(**kwargs)
        total_loss = fm_loss * self.fm + gan_loss * self.gan + mel_loss * self.mel
        out = {
            "fm_loss": fm_loss,
            "gan_loss_g": gan_loss,
            "mel_loss": mel_loss,
            "g_loss": total_loss
        }
        return out