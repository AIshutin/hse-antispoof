import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn


class CELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, target_hat, target, **kwargs):
        return {"loss": self.loss(target_hat, target)}