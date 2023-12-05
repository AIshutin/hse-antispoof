import torch
from torch import Tensor
from antispoof.base.base_metric import BaseMetric
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

# ToDo: precision / accuracy
class Accuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, target, target_hat, **batch):
        return (target == target_hat.argmax(dim=-1)).int().sum() / target_hat.shape[0]