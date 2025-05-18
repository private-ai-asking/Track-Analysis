import abc
from typing import List, Dict

from track_analysis.components.md_common_python.py_common.exceptions.interface_exceptions import \
    InterfaceInstantiationException
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.scrobbling.model.candidate_model import CandidateModel


class CandidateFilterInterface(abc.ABC):
    def __init__(self, logger: HoornLogger, is_child: bool = False):
        self._logger = logger
        self._separator: str = "CandidateFilterInterface"

        if not is_child:
            raise InterfaceInstantiationException(logger, self._separator)

    @abc.abstractmethod
    def filter_candidates(self, candidates: List[CandidateModel], associated_record: Dict) -> List[CandidateModel]:
        """
        Filters candidates based on implementation logic.

        :param candidates: The specific candidates to filter.
        :param associated_record: The associated record, consisting of at least '_n_title', '_n_artist', and '_n_album'.
        :returns: The left-over candidate(s) based on implementation.
        """
