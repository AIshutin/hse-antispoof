import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn


class CELoss(nn.Module):
    def __init__(self, weight=None) -> None:
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight)
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, target_hat, target, **kwargs):
        return {"loss": self.loss(target_hat, target)}