from typing import List, Dict, Optional

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import SimilarityScorer, \
    gaussian_exponential_kernel_confidence_percentage
from track_analysis.components.track_analysis.features.scrobbling.embedding.filtering.candidate_filter_interface import \
    CandidateFilterInterface
from track_analysis.components.track_analysis.features.scrobbling.model.candidate_model import CandidateModel


class BestCandidateSelectorBasedOnTokenSimilarity(CandidateFilterInterface):
    """The default candidate filter used by the NN search."""
    def __init__(self,
                 logger: HoornLogger,
                 similarity_scorer: SimilarityScorer,
                 token_accept_threshold: float,
                 gaussian_sigma: float):
        super().__init__(logger, is_child=True)
        self._separator = "BestCandidateSelectorBasedOnTokenSimilarity"

        self._scorer = similarity_scorer
        self._token_accept_threshold = token_accept_threshold / 100
        self._gaussian_sigma = gaussian_sigma

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def filter_candidates(self, candidates: List[CandidateModel], associated_record: Dict) -> List[CandidateModel]:
        best = self._find_best_candidate(candidates)

        if best is None:
            return []

        best.associated_confidence = gaussian_exponential_kernel_confidence_percentage(
            best.distance, sigma=self._gaussian_sigma
        )

        return [best]

    def _find_best_candidate(self, candidates: List[CandidateModel]) -> Optional[CandidateModel]:
        """
        Return the single candidate to “keep”:
          – if any pass all field thresholds, the one among them with highest combined_token_similarity;
          – otherwise, the one with highest combined_token_similarity overall (fallback).
        """
        # split into those that pass, and the rest
        valid_candidates = [c for c in candidates if self._passes_field_threshold(c)]

        if valid_candidates:
            best: CandidateModel = max(valid_candidates, key=lambda c: c.combined_token_similarity)
            best.passed_demands = True
            return best

        # fallback to overall top-scorer, or None if empty
        best: CandidateModel = max(candidates, key=lambda c: c.combined_token_similarity, default=None)
        best.passed_demands = False
        return best

    def _passes_field_threshold(self, candidate: CandidateModel) -> bool:
        return (
                candidate.title_token_similarity >= self._token_accept_threshold and
                candidate.artist_token_similarity >= self._token_accept_threshold and
                candidate.album_token_similarity >= self._token_accept_threshold
        )
