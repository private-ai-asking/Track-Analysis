from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import SimilarityScorer
from track_analysis.components.track_analysis.features.scrobble_linking.embedding.candidate_retriever_interface import \
    CandidateRetrieverInterface
from track_analysis.components.track_analysis.features.scrobble_linking.embedding.default_candidate_retriever import \
    DefaultCandidateRetriever
from track_analysis.components.track_analysis.features.scrobble_linking.utils.scrobble_data_loader import \
    ScrobbleDataLoader


class CandidateRetrieverFactory:
    """Creates the candidate retriever."""
    def __init__(self, logger: HoornLogger, scrobble_data_loader: ScrobbleDataLoader, scorer: SimilarityScorer):
        self._logger = logger
        self._scrobble_data_loader = scrobble_data_loader
        self._scorer = scorer

    def create(self) -> CandidateRetrieverInterface:
        return DefaultCandidateRetriever(
            logger=self._logger,
            loader=self._scrobble_data_loader,
            token_similarity_scorer=self._scorer,
        )
