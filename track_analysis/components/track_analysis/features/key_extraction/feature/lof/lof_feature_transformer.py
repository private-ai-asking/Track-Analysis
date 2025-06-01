from typing import List

import numpy as np

from track_analysis.components.track_analysis.features.key_extraction.feature.base.feature_transformer import \
    FeatureTransformer


class LOFFeatureTransformer(FeatureTransformer):
    """
    Chromatic histogram to LOF index mapping as in Tanaka's Krumhansl-Schmuckler.
    """
    _lof_idx_map: List[int] = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

    def transform(self, raw: np.ndarray) -> np.ndarray:
        arr = raw.copy()
        return arr[self._lof_idx_map]
