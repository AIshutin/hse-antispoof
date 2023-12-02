from typing import List
from torch import Tensor
from voco.base.base_metric import BaseMetric
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
import cypesq
import logging


class PESQMetric(BaseMetric):
    def __init__(self, fs: int, mode: str = "wb", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(fs, mode).to("cuda")

    def __call__(self, separated, target_wave, target_wave_length, **batch):
        grades = []
        for i in range(target_wave_length.shape[0]):
            try:
                grades.append(
                    self.pesq(separated[i:i+1, :target_wave_length[i]], 
                              target_wave[i:i+1, :target_wave_length[i]]).item()
                )
            except cypesq.NoUtterancesError as exp:
                logging.warning(str(exp)) 
        return sum(grades) / len(grades)