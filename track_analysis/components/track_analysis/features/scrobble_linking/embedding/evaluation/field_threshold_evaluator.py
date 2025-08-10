from typing import List, Dict

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.scrobble_linking.embedding.evaluation.candidate_filter_interface import \
    CandidateEvaluatorInterface
from track_analysis.components.track_analysis.features.scrobble_linking.model.candidate_model import CandidateModel


class FieldThresholdEvaluator(CandidateEvaluatorInterface):
    def __init__(self, logger: HoornLogger, token_accept_threshold: float):
        super().__init__(logger, is_child=True)
        self._separator: str = "FieldThresholdEvaluator"

        self._token_accept_threshold: float = token_accept_threshold

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def evaluate_candidates(self, candidates: List[CandidateModel], associated_record: Dict[str, str]) -> List[
        CandidateModel]:
        for c in candidates:
            c.passed_demands = self._passes_field_threshold(c)

        return candidates

    def _passes_field_threshold(self, candidate: CandidateModel) -> bool:
        return (
                candidate.title_token_similarity >= self._token_accept_threshold and
                candidate.artist_token_similarity >= self._token_accept_threshold and
                candidate.album_token_similarity >= self._token_accept_threshold
        )
