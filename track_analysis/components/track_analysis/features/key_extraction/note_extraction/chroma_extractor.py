import numpy as np
from librosa.feature import chroma_stft

from track_analysis.components.md_common_python.py_common.logging import HoornLogger


class ChromaExtractor:
    def __init__(self, logger: HoornLogger, hop_length: int):
        self.hop_length, self._logger = hop_length, logger
        self._separator: str = "ChromaExtractor"

    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        chroma = chroma_stft(y=audio, sr=sr, hop_length=self.hop_length, norm=None)
        self._logger.debug(f"Chroma shape: {chroma.shape}", separator=self._separator)
        return chroma
