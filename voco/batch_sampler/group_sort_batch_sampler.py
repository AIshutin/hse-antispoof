from torch.utils.data import Sampler, ConcatDataset
from tqdm import tqdm
import numpy as np


class GroupLengthBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.spec_length = []
        datasets = [data_source]
        if isinstance(data_source, ConcatDataset):
            datasets = data_source.datasets
        for d in datasets:
            for el in d._index:
                self.spec_length.append(el['audio_len'])
        self._group()
        print('LENGTH', len(self))

    def _group(self):
        noise = np.random.uniform(-0.1, 0.1, size=len(self.spec_length))
        distorted = []
        for i in range(len(self.spec_length)):
            distorted.append((self.spec_length[i] * (1 + noise[i]), i))
        distorted.sort()
        self.order = [el[1] for el in distorted]
        self.i = 0

    def __iter__(self):
        self._group()
        while self.i <= len(self.order):
            idx = self.order[self.i:self.i+self.batch_size]
            yield idx
            self.i += self.batch_size

    def __len__(self):
        return ((len(self.order) - 1) // self.batch_size) + 1
