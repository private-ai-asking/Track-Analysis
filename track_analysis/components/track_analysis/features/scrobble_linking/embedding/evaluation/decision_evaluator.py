from typing import List, Dict

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.scrobble_linking.embedding.evaluation.candidate_filter_interface import \
    CandidateEvaluatorInterface
from track_analysis.components.track_analysis.features.scrobble_linking.model.candidate_model import CandidateModel, \
    DecisionBin


class DecisionEvaluator(CandidateEvaluatorInterface):
    """The default candidate filter used by the NN search."""
    def __init__(self,
                 logger: HoornLogger,
                 confidence_accept_threshold: float,
                 confidence_reject_threshold: float,
                 token_accept_threshold: float):
        super().__init__(logger, is_child=True)
        self._separator = "DecisionEvaluator"

        self._confidence_accept_threshold = confidence_accept_threshold
        self._confidence_reject_threshold = confidence_reject_threshold
        self._token_accept_threshold = token_accept_threshold

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def evaluate_candidates(self, candidates: List[CandidateModel], associated_record: Dict) -> List[CandidateModel]:
        for c in candidates:
            c.decision_bin = self._decision(c.associated_confidence, c.combined_token_similarity, c.passed_demands)

            if c.decision_bin in [DecisionBin.UNCERTAIN, DecisionBin.REJECT]:
                c.associated_confidence = 0.0

        return candidates

    def _decision(self, confidence: float, token_sim: float, passed_demands: bool) -> DecisionBin:
        # 1. Compute clear, descriptive boolean checks
        is_confident = confidence >= self._confidence_accept_threshold
        is_low_confidence = confidence <= self._confidence_reject_threshold
        is_token_similar = token_sim >= self._token_accept_threshold

        # 2. Define decision states with at most two-part expressions
        accept = is_confident and is_token_similar and passed_demands
        reject = (is_confident and not is_token_similar) or is_low_confidence

        # 3. Warn only if accept and reject both True (impossible for uncertain)
        if accept and reject:
            self._logger.warning(
                f"Conflicting decision states: accept={accept}, reject={reject}",
                separator=self._separator
            )

        # 4. Early return for primary cases
        if accept:
            return DecisionBin.ACCEPT
        if reject:
            return DecisionBin.REJECT

        # 5. Default (typical) case
        return DecisionBin.UNCERTAIN
