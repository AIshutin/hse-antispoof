import torch
import torchaudio


class STFT:
    def __init__(self, n_fft, hop_length, window_length, **kwargs) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_length = window_length
    
    def __call__(self, X):
        return torch.stft(X, self.n_fft, 
                          return_complex=True, 
                          hop_length=self.hop_length, 
                          win_length=self.window_length).real


class STFT2:
    def __init__(self, n_fft, hop_length, window_length, **kwargs) -> None:
        self.to_spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window_length,
        )
    
    def __call__(self, X):
        spec = self.to_spec(X)
        return torch.log(spec + 1)


class STFT3:
    def __init__(self, n_fft, hop_length, window_length, **kwargs) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_length = window_length
    
    def __call__(self, X):
        return torch.log(torch.abs(torch.stft(X, self.n_fft, 
                          return_complex=True, 
                          hop_length=self.hop_length, 
                          win_length=self.window_length).real) + 1e-9)