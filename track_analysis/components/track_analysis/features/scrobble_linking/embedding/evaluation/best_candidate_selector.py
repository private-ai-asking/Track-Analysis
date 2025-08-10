from typing import List, Dict, Optional

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.scrobble_linking.embedding.evaluation.candidate_filter_interface import \
    CandidateEvaluatorInterface
from track_analysis.components.track_analysis.features.scrobble_linking.model.candidate_model import CandidateModel


class BestCandidateSelector(CandidateEvaluatorInterface):
    """The default candidate filter used by the NN search."""
    def __init__(self, logger: HoornLogger):
        super().__init__(logger, is_child=True)
        self._separator = "BestCandidateSelector"
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def evaluate_candidates(self, candidates: List[CandidateModel], associated_record: Dict) -> List[CandidateModel]:
        best = self._find_best_candidate(candidates)

        if best is None:
            return []

        return [best]

    def _find_best_candidate(self, candidates: List[CandidateModel]) -> Optional[CandidateModel]:
        """
        Return the single candidate to “keep”:
          – if any pass all field thresholds, the one among them with highest combined_token_similarity;
          – otherwise, the one with highest combined_token_similarity overall (fallback).
        """
        # split into those that pass, and the rest
        valid_candidates = [c for c in candidates if c.passed_demands]

        if valid_candidates:
            best: CandidateModel = max(valid_candidates, key=lambda c: c.combined_token_similarity)
            self._logger.debug(f"Found valid candidate: {best.uuid}", separator=self._separator)
            return best

        # fallback to overall top-scorer, or None if empty
        best: CandidateModel = max(candidates, key=lambda c: c.combined_token_similarity, default=None)
        self._logger.debug(f"Did not find valid candidate: {best.uuid}", separator=self._separator)
        return best
