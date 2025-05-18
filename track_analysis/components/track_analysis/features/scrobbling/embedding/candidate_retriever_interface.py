import abc
from typing import Dict, List

import numpy

from track_analysis.components.md_common_python.py_common.exceptions.interface_exceptions import \
    InterfaceInstantiationException
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.scrobbling.model.candidate_model import CandidateModel
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_data_loader import ScrobbleDataLoader


class CandidateRetrieverInterface(abc.ABC):
    def __init__(self, logger: HoornLogger, loader: ScrobbleDataLoader, is_child: bool = False):
        self._logger = logger
        self._loader = loader
        self._separator: str = "CandidateRetrieverInterface"

        if not is_child:
            raise InterfaceInstantiationException(logger, self._separator)

    @abc.abstractmethod
    def retrieve_candidates(self, record: Dict, neighbour_indices: List[int], distances: numpy.array) -> List[CandidateModel]:
        """
        Retrieves candidates based on the parameters given as processed by the specific implementation.
        :param record: The current track's record consisting of at least '_n_title', '_n_artist', and '_n_album'
        :param neighbour_indices: Faiss Index indices for the Neighbor rows.
        :param distances: Distances to each neighbor.
        :return:
        """
