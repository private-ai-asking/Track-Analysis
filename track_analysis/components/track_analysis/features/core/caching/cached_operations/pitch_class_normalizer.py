import pprint
from pathlib import Path

import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.core.caching.cached_operations.shared import MEMORY


@MEMORY.cache(identifier_arg="file_path", ignore=["pitch_classes"])
def _normalize(file_path: Path, pitch_classes: np.ndarray) -> np.ndarray:
    eps = 1e-12
    norms_l2 = np.linalg.norm(pitch_classes, ord=2, axis=0, keepdims=True)  # shape (1, n_frames)
    X_l2 = pitch_classes / (norms_l2 + eps)
    return X_l2

class PitchClassesNormalizer:
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def normalize_pitch_classes(self, file_path: Path, pitch_classes: np.ndarray) -> np.ndarray:
        normalized_pitch_classes = _normalize(file_path, pitch_classes)

        self._logger.debug(f"Normalized Pitch Classes Shape: {normalized_pitch_classes.shape}", separator=self._separator)
        self._logger.debug(f"Normalized Pitch Classes:\n{pprint.pformat(normalized_pitch_classes)}", separator=self._separator)

        return normalized_pitch_classes
