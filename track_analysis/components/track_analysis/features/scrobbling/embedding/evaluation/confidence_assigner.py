from typing import List, Dict

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import gaussian_exponential_kernel_confidence_percentage
from track_analysis.components.track_analysis.features.scrobbling.embedding.evaluation.candidate_filter_interface import \
    CandidateEvaluatorInterface
from track_analysis.components.track_analysis.features.scrobbling.model.candidate_model import CandidateModel


class GaussianConfidenceAssigner(CandidateEvaluatorInterface):
    """The default candidate filter used by the NN search."""
    def __init__(self, logger: HoornLogger, gaussian_sigma: float):
        super().__init__(logger, is_child=True)
        self._separator = "GaussianConfidenceAssigner"

        self._gaussian_sigma: float = gaussian_sigma

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def evaluate_candidates(self, candidates: List[CandidateModel], associated_record: Dict) -> List[CandidateModel]:
        for c in candidates:
            c.associated_confidence = gaussian_exponential_kernel_confidence_percentage(
                c.distance, sigma=self._gaussian_sigma
            )

        return candidates
