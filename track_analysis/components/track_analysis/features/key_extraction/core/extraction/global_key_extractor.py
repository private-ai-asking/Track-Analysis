from typing import List, Tuple

import numpy as np

from track_analysis.components.md_common_python.py_common.algorithms.similarity import SimilarityMatcher
from track_analysis.components.md_common_python.py_common.logging import HoornLogger


class GlobalKeyEstimator:
    """
    Given feature vectors and corresponding intervals (durations), compute
    a single global chroma vector (weighted by segment duration), and pick
    the best global key label.
    """
    def __init__(self, logger: HoornLogger, local_templates: dict[str, np.ndarray]):
        self._logger = logger
        self._separator = self.__class__.__name__

        # To pick a global key, we re‐use the same template set as local:
        self._global_matcher = SimilarityMatcher(
            logger=logger,
            templates=local_templates,
            label_order=list(local_templates.keys()),
            verbose=False,
        )
        self._logger.info("Initialized GlobalKeyEstimator.", separator=self._separator)

    def estimate_global_key(
            self,
            feature_matrix: List[np.ndarray],
            intervals: List[Tuple[float, float]],
    ) -> str:
        """
        1) Compute each segment's duration (end_frame - start_frame),
        2) Build one global chroma by weighting each feature vector by its duration,
        3) Normalize the global chroma (L1‐norm),
        4) Run the global_similarity match and pick the top label.
        """
        self._logger.info("Starting global key estimation.", separator=self._separator)

        # 1. Compute durations
        durations = np.array([end - start for (start, end) in intervals], dtype=float)

        # 2. Weighted sum of all feature vectors
        global_chroma = np.zeros(12, dtype=float)
        for i, vec in enumerate(feature_matrix):
            global_chroma += durations[i] * vec

        # 3. Normalize to L1‐norm
        norm = np.linalg.norm(global_chroma, ord=1)
        if norm > 0.0:
            global_chroma /= norm

        # 4. Match & pick best label
        score_result = self._global_matcher.match([global_chroma])
        scores = score_result.matrix[0]
        labels = score_result.labels
        best_idx = int(np.argmax(scores))
        best_label = labels[best_idx]

        self._logger.info(f"Global key estimated: {best_label}", separator=self._separator)
        return best_label
