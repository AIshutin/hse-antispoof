import logging
from pathlib import Path
import torchaudio
from antispoof.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir):
        index = []
        for path in Path(audio_dir).iterdir():
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry = {}
                entry["path"] = str(path)
                t_info = torchaudio.info(entry["path"])
                entry["audio_len"] = t_info.num_frames / t_info.sample_rate
                index.append(entry)
        super().__init__(index)