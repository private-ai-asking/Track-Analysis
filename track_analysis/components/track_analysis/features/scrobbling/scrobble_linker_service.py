from pathlib import Path

import pandas as pd
from pandas import DataFrame
from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import StringUtils, SimilarityScorer
from track_analysis.components.track_analysis.features.scrobbling.embedding.embedding_builder import EmbeddingBuilder
from track_analysis.components.track_analysis.features.scrobbling.embedding.embedding_searcher import EmbeddingSearcher
from track_analysis.components.track_analysis.features.scrobbling.model.scrabble_cache_algorithm_parameters import \
    ScrobbleCacheAlgorithmParameters
from track_analysis.components.track_analysis.features.scrobbling.scrobble_cache_builder import ScrobbleCacheBuilder
from track_analysis.components.track_analysis.features.scrobbling.scrobble_matcher import ScrobbleMatcher
from track_analysis.components.track_analysis.features.scrobbling.utils.cache_helper import ScrobbleCacheHelper
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_data_loader import ScrobbleDataLoader
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_utility import ScrobbleUtility
from track_analysis.components.track_analysis.library.configuration.model.configuration import \
    TrackAnalysisConfigurationModel


class ScrobbleLinkerService:
    """Used to link scrobbles with library-level data using fuzzy matching. High-level API."""

    def __init__(self,
                 logger: HoornLogger,
                 data_loader: ScrobbleDataLoader,
                 string_utils: StringUtils,
                 embedder: SentenceTransformer,
                 scrobble_utils: ScrobbleUtility,
                 cache_builder: CacheBuilder,
                 cache_helper: ScrobbleCacheHelper,
                 embedding_searcher: EmbeddingSearcher,
                 scorer: SimilarityScorer,
                 app_config: TrackAnalysisConfigurationModel):
        self._logger: HoornLogger = logger
        self._separator: str = "ScrobbleLinker"
        self._string_utils: StringUtils = string_utils
        self._combo_key: str = app_config.scrobble_linker.field_combination_key
        self._scrobble_data_loader: ScrobbleDataLoader = data_loader
        self._keys_path: Path = app_config.paths.library_keys
        self._minimum_confidence_threshold: float = app_config.scrobble_linker.token_accept_threshold
        self._config = app_config

        cache_builder: CacheBuilder = cache_builder

        self._scrobble_matcher: ScrobbleMatcher = ScrobbleMatcher(
            logger,
            cache_builder,
            scrobble_utils,
            data_loader=self._scrobble_data_loader,
            app_config=app_config
        )

        self._embedding_builder: EmbeddingBuilder = EmbeddingBuilder(
            logger,
            self._scrobble_data_loader,
            sample_scrobbles=app_config.additional_config.test_sample_size,
            scrobble_utils=scrobble_utils,
            track_analysis_config=app_config
        )

        self._scrobble_cache_builder: ScrobbleCacheBuilder = ScrobbleCacheBuilder(
            logger,
            cache_builder,
            data_loader,
            sample_size=app_config.additional_config.test_sample_size,
            scrobble_utils=scrobble_utils,
            keys_path=app_config.paths.library_keys,
            embedding_model=embedder,
            test=app_config.development.test_cache_builder_mode,
            parameters=ScrobbleCacheAlgorithmParameters(
                gold_standard_csv_path=app_config.paths.gold_standard_data,
                embed_weights=app_config.scrobble_linker.embedding_weights,
                manual_override_path=app_config.paths.manual_override,
                token_accept_threshold=app_config.scrobble_linker.token_accept_threshold,
            ),
            cache_helper=cache_helper,
            manual_override_path=app_config.paths.manual_override,
            searcher=embedding_searcher,
            scorer=scorer,
            app_config=app_config
        )

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def build_embeddings_for_library(self) -> None:
        self._embedding_builder.build_embeddings()

    def build_cache(self) -> None:
        self._scrobble_cache_builder.build_cache()

    def test_parameters(self) -> None:
        self._scrobble_cache_builder.test_parameters()

    def link_scrobbles(self) -> pd.DataFrame:
        """Links scrobble data to library data by matching tracks and writing the associated Track ID
        (if any) into the enriched scrobble data csv."""
        self._logger.info("Starting to link scrobbles...", separator=self._separator)

        scrobble_data: DataFrame = self._scrobble_data_loader.get_scrobble_data()

        enriched_scrobble_data: DataFrame = self._scrobble_matcher.link_scrobbles(scrobble_data)
        self._log_unmatched_amount(enriched_scrobble_data)
        return enriched_scrobble_data

    def _log_unmatched_amount(self, enriched_scrobble_date: pd.DataFrame) -> None:
        # Log unmatched count
        unmatched: int = enriched_scrobble_date["track_uuid"].eq(self._config.additional_config.no_match_label).sum()
        total: int = len(enriched_scrobble_date)
        self._logger.info(
            f"Linked {total - unmatched} of {total} scrobbles. {unmatched} remain unmatched.",
            separator=self._separator
        )
