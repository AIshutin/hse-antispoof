import torch
import torchaudio


class LFCC:
    def __init__(self, n_dim, sr, speckwargs, **kwargs) -> None:
        assert(n_dim % 3 == 0)
        self.lfcc = torchaudio.transforms.LFCC(sr, n_lfcc=n_dim // 3, speckwargs=speckwargs)
        self.delta = torchaudio.transforms.ComputeDeltas()
    
    def __call__(self, X):
        X = self.lfcc(X)
        Xd = self.delta(X)
        Xdd = self.delta(Xd)
        X = torch.cat((X, Xd, Xdd), dim=1)
        return X


class LFCC2:
    def __init__(self, n_dim, sr, speckwargs, **kwargs) -> None:
        assert(n_dim % 3 == 0)
        self.lfcc = torchaudio.transforms.LFCC(sr, n_lfcc=n_dim, speckwargs=speckwargs)
    
    def __call__(self, X):
        X = self.lfcc(X)
        return X