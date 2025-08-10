from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import SimilarityScorer
from track_analysis.components.track_analysis.features.scrobble_linking.embedding.candidate_retriever_interface import \
    CandidateRetrieverInterface
from track_analysis.components.track_analysis.features.scrobble_linking.embedding.embedding_searcher import \
    EmbeddingSearcher
from track_analysis.components.track_analysis.features.scrobble_linking.factory.candidate_retriever_factory import \
    CandidateRetrieverFactory
from track_analysis.components.track_analysis.features.scrobble_linking.utils.scrobble_data_loader import \
    ScrobbleDataLoader
from track_analysis.components.track_analysis.features.scrobble_linking.utils.scrobble_utility import ScrobbleUtility


class EmbeddingSearcherFactory:
    """Creates the embedding searcher."""
    def __init__(self, logger: HoornLogger,
                 scrobble_data_loader: ScrobbleDataLoader, similarity_scorer: SimilarityScorer,
                 scrobble_utils: ScrobbleUtility):
        self._logger = logger
        self._scrobble_data_loader = scrobble_data_loader
        self._similarity_scorer = similarity_scorer
        self._scrobble_utils = scrobble_utils

        self._candidate_retriever = self._get_candidate_retriever()

    def _get_candidate_retriever(self) -> CandidateRetrieverInterface:
        factory = CandidateRetrieverFactory(self._logger, self._scrobble_data_loader, self._similarity_scorer)
        return factory.create()

    def create(self, top_k: int = 5) -> EmbeddingSearcher:
        return EmbeddingSearcher(
            self._logger,
            top_k=top_k,
            loader=self._scrobble_data_loader,
            utility=self._scrobble_utils,
            candidate_retriever=self._candidate_retriever
        )
