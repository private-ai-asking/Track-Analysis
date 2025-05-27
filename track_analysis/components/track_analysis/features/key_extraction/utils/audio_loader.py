from pathlib import Path
import librosa
import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger


class AudioLoader:
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator: str = "AudioLoader"

    def load(self, path: Path) -> tuple[np.ndarray,int]:
        audio, sr = librosa.load(path, sr=None)
        self._logger.debug(f"Loaded {len(audio)} samples @ {sr}Hz", separator=self._separator)
        return audio, sr
