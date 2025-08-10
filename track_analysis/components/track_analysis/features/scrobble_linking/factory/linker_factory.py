from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import StringUtils, SimilarityScorer
from track_analysis.components.track_analysis.features.scrobble_linking.embedding.embedding_searcher import \
    EmbeddingSearcher
from track_analysis.components.track_analysis.features.scrobble_linking.scrobble_linker_service import \
    ScrobbleLinkerService
from track_analysis.components.track_analysis.features.scrobble_linking.utils.cache_helper import ScrobbleCacheHelper
from track_analysis.components.track_analysis.features.scrobble_linking.utils.scrobble_data_loader import \
    ScrobbleDataLoader
from track_analysis.components.track_analysis.features.scrobble_linking.utils.scrobble_utility import ScrobbleUtility
from track_analysis.components.track_analysis.library.configuration.model.configuration import \
    TrackAnalysisConfigurationModel


class ScrobbleLinkerFactory:
    """Creates a scrobble linker service."""
    def __init__(self,
                 logger: HoornLogger,
                 string_utils: StringUtils,
                 scrobble_util: ScrobbleUtility,
                 embedder: SentenceTransformer,
                 embedding_searcher: EmbeddingSearcher,
                 similarity_scorer: SimilarityScorer,
                 cache_helper: ScrobbleCacheHelper,
                 cache_builder: CacheBuilder,
                 data_loader: ScrobbleDataLoader,
                 app_configuration: TrackAnalysisConfigurationModel,
                 ):
        self._logger = logger
        self._string_utils = string_utils
        self._scrobble_util = scrobble_util
        self._embedder = embedder
        self._embedding_searcher = embedding_searcher
        self._similarity_scorer = similarity_scorer
        self._cache_helper = cache_helper
        self._cache_builder = cache_builder
        self._app_configuration = app_configuration
        self._scrobble_data_loader = data_loader


    def create(self) -> ScrobbleLinkerService:
        return ScrobbleLinkerService(
            self._logger,
            data_loader=self._scrobble_data_loader,
            string_utils=self._string_utils,
            embedder=self._embedder,
            scrobble_utils=self._scrobble_util,
            cache_builder=self._cache_builder,
            cache_helper=self._cache_helper,
            embedding_searcher=self._embedding_searcher,
            scorer=self._similarity_scorer,
            app_config=self._app_configuration,
        )
