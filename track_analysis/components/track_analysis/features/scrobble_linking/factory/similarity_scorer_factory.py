from typing import Dict

from rapidfuzz import fuzz

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import SimilarityScorer
from track_analysis.components.md_common_python.py_common.utils.data_analysis.comparison_utils import SimilarityFunc


class SimilarityScorerFactory:
    """Creates the similarity scorer."""
    def __init__(self, logger: HoornLogger):
        self._logger = logger

    def create(self,
               embed_weights: Dict[str, float], token_accept_threshold: float,
               similarity_func: SimilarityFunc = fuzz.token_sort_ratio) -> SimilarityScorer:
        return SimilarityScorer(
            embed_weights,
            logger=self._logger,
            threshold=token_accept_threshold,
            similarity_func=similarity_func,
        )
