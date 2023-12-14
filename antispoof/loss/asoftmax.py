import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn


class ASoftmaxLoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, p0, p1, target, **kwargs):
        p = p0 * (1 - target) + p1 * target
        loss = -torch.log(p).mean()
        return {"loss": loss}