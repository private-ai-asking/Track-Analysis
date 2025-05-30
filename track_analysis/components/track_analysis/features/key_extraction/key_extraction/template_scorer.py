from typing import Dict, List, Callable, Optional

import numpy as np


# noinspection PyTypeChecker
class TemplateScorer:
    def __init__(self, templates: Dict[str, np.ndarray], similarity_fn: Optional[Callable[[np.ndarray, np.ndarray], float]]=None):
        """
        templates: mapping from state name to normalized template vector
        similarity_fn: function(v1, v2) -> float
        """
        self._templates = templates
        self._similarity_fn = similarity_fn or self._pearson_corr

    @staticmethod
    def _pearson_corr(v1: np.ndarray, v2: np.ndarray) -> float:
        c = np.corrcoef(v1, v2)
        return c[0, 1]

    def score(self, feature_vectors: List[np.ndarray]) -> np.ndarray:
        n = len(feature_vectors)
        m = len(self._templates)
        raw_scores = np.zeros((n, m))
        keys = list(self._templates.keys())
        for i, fv in enumerate(feature_vectors):
            for j, key in enumerate(keys):
                raw_scores[i, j] = self._similarity_fn(fv, self._templates[key])
        return raw_scores, keys
