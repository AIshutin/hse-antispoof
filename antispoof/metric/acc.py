import torch
from torch import Tensor
from antispoof.base.base_metric import BaseMetric

# ToDo: precision / accuracy
class Accuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, target, p0, p1, **kwargs):
        target_hat = (p1 > p0).int()
        return (target == target_hat).int().sum() / target_hat.shape[0]