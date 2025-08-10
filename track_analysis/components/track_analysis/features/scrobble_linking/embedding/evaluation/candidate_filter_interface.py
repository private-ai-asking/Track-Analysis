import abc
from typing import List, Dict

from track_analysis.components.md_common_python.py_common.exceptions.interface_exceptions import \
    InterfaceInstantiationException
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.scrobble_linking.model.candidate_model import CandidateModel


class CandidateEvaluatorInterface(abc.ABC):
    def __init__(self, logger: HoornLogger, is_child: bool = False):
        self._logger = logger
        self._separator: str = "CandidateFilterInterface"

        if not is_child:
            raise InterfaceInstantiationException(logger, self._separator)

    @abc.abstractmethod
    def evaluate_candidates(self, candidates: List[CandidateModel], associated_record: Dict[str, str]) -> List[CandidateModel]:
        """
        Evaluates candidates based on implementation logic.
        This can be adding metadata or filtering the list down.

        :param candidates: The specific candidates to evaluate.
        :param associated_record: The associated record, consisting of at least '_n_title', '_n_artist', and '_n_album'.
        :returns: The left-over candidate(s) based on implementation.
        """
