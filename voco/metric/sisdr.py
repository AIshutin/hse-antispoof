import torch
from torch import Tensor
from voco.base.base_metric import BaseMetric
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


class SISDR(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sisdr = ScaleInvariantSignalDistortionRatio().to("cuda")

    def __call__(self, separated, target_wave, target_wave_length, **batch):
        grades = []
        for i in range(target_wave_length.shape[0]):
            grades.append(
                self.sisdr(separated[i:i+1, :target_wave_length[i]], 
                          target_wave[i:i+1, :target_wave_length[i]]).item()
            )
        return sum(grades) / len(grades)