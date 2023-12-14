import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn


class CELoss(nn.Module):
    def __init__(self, weight=None) -> None:
        super().__init__()
        print('LOSS WEIGHT', weight)
        if weight is not None and len(weight) != 0:
            weight = torch.tensor(weight).float()
        else:
            weight = None
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, target, **kwargs):
        return {"loss": self.loss(logits, target)}