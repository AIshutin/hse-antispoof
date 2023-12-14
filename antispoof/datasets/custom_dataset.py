import logging
from pathlib import Path
import torchaudio
from antispoof.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, path, **kwargs):
        index = []
        for audio in sorted(Path(path).iterdir()):
            if audio.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                try:
                    t_info = torchaudio.info(audio)
                    length = t_info.num_frames / t_info.sample_rate
                except:
                    length = 10
                index.append({
                    "audio": str(audio),
                    "target": -1,
                    "cls": 'unknown',
                    "audio_length": length
                })
        super().__init__(index, **kwargs)