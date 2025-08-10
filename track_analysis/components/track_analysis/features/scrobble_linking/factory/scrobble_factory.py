from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import StringUtils, SimilarityScorer
from track_analysis.components.track_analysis.features.scrobble_linking.embedding.embedding_searcher import \
    EmbeddingSearcher
from track_analysis.components.track_analysis.features.scrobble_linking.factory.cache_builder_factory import \
    CacheBuilderFactory
from track_analysis.components.track_analysis.features.scrobble_linking.factory.cache_helper_factory import \
    CacheHelperFactory
from track_analysis.components.track_analysis.features.scrobble_linking.factory.data_loader_factory import \
    ScrobbleDataLoaderFactory
from track_analysis.components.track_analysis.features.scrobble_linking.factory.embedder_factory import EmbedderFactory
from track_analysis.components.track_analysis.features.scrobble_linking.factory.embedding_searcher_factory import \
    EmbeddingSearcherFactory
from track_analysis.components.track_analysis.features.scrobble_linking.factory.linker_factory import \
    ScrobbleLinkerFactory
from track_analysis.components.track_analysis.features.scrobble_linking.factory.similarity_scorer_factory import \
    SimilarityScorerFactory
from track_analysis.components.track_analysis.features.scrobble_linking.factory.utility_factory import \
    ScrobbleUtilityFactory
from track_analysis.components.track_analysis.features.scrobble_linking.get_unmatched_library_tracks import \
    UnmatchedLibraryTracker
from track_analysis.components.track_analysis.features.scrobble_linking.processor.uncertain_keys_processor import \
    UncertainKeysProcessor
from track_analysis.components.track_analysis.features.scrobble_linking.scrobble_linker_service import \
    ScrobbleLinkerService
from track_analysis.components.track_analysis.features.scrobble_linking.utils.cache_helper import ScrobbleCacheHelper
from track_analysis.components.track_analysis.features.scrobble_linking.utils.scrobble_data_loader import \
    ScrobbleDataLoader
from track_analysis.components.track_analysis.features.scrobble_linking.utils.scrobble_utility import ScrobbleUtility
from track_analysis.components.track_analysis.library.configuration.model.configuration import \
    TrackAnalysisConfigurationModel


class ScrobbleFactory:
    """High-Level Factory to use to get scrobble linking related components."""
    def __init__(self,
                 logger: HoornLogger,
                 string_utils: StringUtils,
                 app_configuration: TrackAnalysisConfigurationModel):
        self._logger = logger
        self._string_utils = string_utils

        self._embedder = self._get_embedder(app_configuration)
        self._scrobble_utils = self._get_scrobble_utility(app_configuration)
        self._scrobble_data_loader = self._get_scrobble_data_loader(app_configuration)
        self._cache_builder = self._get_cache_builder(app_configuration)
        self._cache_helper = self._get_cache_helper()
        self._similarity_scorer = self._get_similarity_scorer(app_configuration)
        self._embedding_searcher = self._get_embedding_searcher()

        self._configuration = app_configuration

    @staticmethod
    def _get_embedder(app_configuration: TrackAnalysisConfigurationModel) -> SentenceTransformer:
        embedder_factory: EmbedderFactory = EmbedderFactory(app_configuration.scrobble_linker.embedder_path)
        return embedder_factory.create()

    def _get_scrobble_utility(self, app_configuration: TrackAnalysisConfigurationModel) -> ScrobbleUtility:
        scrobble_utility_factory = ScrobbleUtilityFactory(self._logger, self._embedder)

        return scrobble_utility_factory.create(
            app_configuration.scrobble_linker.embedding_weights,
            app_configuration.scrobble_linker.embedding_batch_size,
            app_configuration.scrobble_linker.field_combination_key
        )

    def _get_scrobble_data_loader(self, app_configuration: TrackAnalysisConfigurationModel) -> ScrobbleDataLoader:
        scrobble_data_loader_factory = ScrobbleDataLoaderFactory(
            self._logger,
            self._string_utils,
            self._scrobble_utils,
            app_configuration.paths.library_data,
            app_configuration.scrobble_linker.scrobble_index_dir,
            app_configuration.paths.library_keys,
            app_configuration.paths.scrobble_data
        )
        return scrobble_data_loader_factory.create()

    def _get_cache_builder(self, app_configuration: TrackAnalysisConfigurationModel) -> CacheBuilder:
        cache_builder_factory = CacheBuilderFactory(self._logger, app_configuration.paths.scrobble_cache)
        return cache_builder_factory.create(tree_separator=app_configuration.scrobble_linker.field_combination_key)

    def _get_cache_helper(self) -> ScrobbleCacheHelper:
        cache_helper_factory = CacheHelperFactory(self._logger, self._scrobble_data_loader, self._cache_builder)
        return cache_helper_factory.create()

    def _get_similarity_scorer(self, app_configuration: TrackAnalysisConfigurationModel) -> SimilarityScorer:
        factory = SimilarityScorerFactory(self._logger)
        return factory.create(app_configuration.scrobble_linker.embedding_weights, app_configuration.scrobble_linker.token_accept_threshold)

    def _get_embedding_searcher(self) -> EmbeddingSearcher:
        factory = EmbeddingSearcherFactory(self._logger, self._scrobble_data_loader, self._similarity_scorer, self._scrobble_utils)
        return factory.create()

    def create_scrobble_linker_service(self) -> ScrobbleLinkerService:
        factory = ScrobbleLinkerFactory(
            self._logger,
            self._string_utils,
            self._scrobble_utils,
            self._embedder,
            self._embedding_searcher,
            self._similarity_scorer,
            self._cache_helper,
            self._cache_builder,
            self._scrobble_data_loader,
            self._configuration
        )
        return factory.create()

    def create_uncertain_keys_processor(self) -> UncertainKeysProcessor:
        return UncertainKeysProcessor(self._logger, self._embedding_searcher, self._scrobble_utils, self._scrobble_data_loader, self._configuration)

    def create_unmatched_library_tracker(self) -> UnmatchedLibraryTracker:
        return UnmatchedLibraryTracker(self._logger, self._scrobble_data_loader, self._configuration.paths.scrobble_cache)
