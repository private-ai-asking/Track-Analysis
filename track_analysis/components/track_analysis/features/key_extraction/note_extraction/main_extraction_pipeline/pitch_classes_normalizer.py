import os
import pprint
from pathlib import Path

import numpy as np
from joblib import Memory

from track_analysis.components.md_common_python.py_common.logging import HoornLogger

def _normalize(pitch_classes: np.ndarray) -> np.ndarray:
    eps = 1e-12
    norms_l2 = np.linalg.norm(pitch_classes, ord=2, axis=0, keepdims=True)  # shape (1, n_frames)
    X_l2 = pitch_classes / (norms_l2 + eps)
    return X_l2

class PitchClassesNormalizer:
    def __init__(self, logger: HoornLogger, cache_dir: Path):
        self._logger = logger
        self._separator = self.__class__.__name__

        os.makedirs(cache_dir, exist_ok=True)
        self._normalize = Memory(cache_dir, verbose=0).cache(_normalize)

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def normalize_pitch_classes(self, pitch_classes: np.ndarray) -> np.ndarray:
        normalized_pitch_classes = self._normalize(pitch_classes)

        self._logger.debug(f"Normalized Pitch Classes Shape: {normalized_pitch_classes.shape}", separator=self._separator)
        self._logger.debug(f"Normalized Pitch Classes:\n{pprint.pformat(normalized_pitch_classes)}", separator=self._separator)

        return normalized_pitch_classes
