from typing import Protocol

import numpy as np
from skimage.filters import threshold_otsu

from track_analysis.components.md_common_python.py_common.logging import HoornLogger


class ThresholdingStrategy(Protocol):
    def threshold(self, x: np.ndarray) -> float: ...

class OtsuThresholding:
    @staticmethod
    def threshold(x: np.ndarray) -> float:
        return threshold_otsu(x)

class BinaryMaskGenerator:
    def __init__(self, strategy: ThresholdingStrategy, logger: HoornLogger):
        self.strategy, self._logger = strategy, logger
        self._separator: str = "BinaryMaskGenerator"

    def binarize(self, chroma_db: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(chroma_db, dtype=bool)
        for pc in range(chroma_db.shape[0]):
            thr = self.strategy.threshold(chroma_db[pc])
            mask[pc] = chroma_db[pc] >= thr
            self._logger.debug(f"PC {pc}: thr={thr:.1f}dB, {mask[pc].sum()} frames", separator=self._separator)
        return mask
