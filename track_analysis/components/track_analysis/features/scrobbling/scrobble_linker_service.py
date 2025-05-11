from pathlib import Path

import pandas as pd
from pandas import DataFrame
from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import StringUtils, SimilarityScorer
from track_analysis.components.track_analysis.constants import CACHE_DIRECTORY, CLEAR_CACHE, TEST_SAMPLE_SIZE, \
    NO_MATCH_LABEL, TEST_CACHE_BUILDER_MODE
from track_analysis.components.track_analysis.features.scrobbling.embedding_builder import EmbeddingBuilder
from track_analysis.components.track_analysis.features.scrobbling.scrobble_cache_builder import ScrobbleCacheBuilder
from track_analysis.components.track_analysis.features.scrobbling.scrobble_data_loader import ScrobbleDataLoader
from track_analysis.components.track_analysis.features.scrobbling.scrobble_matcher import ScrobbleMatcher
from track_analysis.components.track_analysis.features.scrobbling.scrobble_utility import ScrobbleUtility


class ScrobbleLinkerService:
    """Used to link scrobbles with library-level data using fuzzy matching. High-level API."""

    def __init__(self,
                 logger: HoornLogger,
                 data_loader: ScrobbleDataLoader,
                 string_utils: StringUtils,
                 embedder: SentenceTransformer,
                 keys_path: Path,
                 index_path: Path,
                 scrobble_utils: ScrobbleUtility,
                 combo_key: str = "||",
                 minimum_confidence_threshold: float = 90.0):
        self._logger: HoornLogger = logger
        self._separator: str = "ScrobbleLinker"
        self._string_utils: StringUtils = string_utils
        self._combo_key: str = combo_key
        self._embedder: SentenceTransformer = embedder
        self._scrobble_data_loader: ScrobbleDataLoader = data_loader
        self._keys_path: Path = keys_path
        self._index_path: Path = index_path
        self._minimum_confidence_threshold: float = minimum_confidence_threshold

        cache_path: Path = CACHE_DIRECTORY.joinpath("scrobble_cache.json")

        if CLEAR_CACHE:
            cache_path.unlink(missing_ok=True)

        cache_builder: CacheBuilder = CacheBuilder(logger, cache_path, tree_separator=self._combo_key)

        self._scrobble_matcher: ScrobbleMatcher = ScrobbleMatcher(
            logger,
            cache_builder,
            scrobble_utils
        )

        self._embedding_builder: EmbeddingBuilder = EmbeddingBuilder(
            logger,
            self._scrobble_data_loader,
            embedder=embedder,
            combo_key=self._combo_key,
            sample_scrobbles=TEST_SAMPLE_SIZE
        )

        self._scrobble_cache_builder: ScrobbleCacheBuilder = ScrobbleCacheBuilder(
            logger,
            cache_builder,
            data_loader,
            sample_size=TEST_SAMPLE_SIZE,
            scrobble_utils=scrobble_utils,
            index_path=index_path,
            keys_path=keys_path,
            embedding_model=embedder,
            test=TEST_CACHE_BUILDER_MODE
        )

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def build_embeddings_for_library(self) -> None:
        self._embedding_builder.build_embeddings()

    def build_cache(self) -> None:
        self._scrobble_cache_builder.build_cache()

    def link_scrobbles(self) -> pd.DataFrame:
        """Links scrobble data to library data by matching tracks and writing the associated Track ID
        (if any) into the enriched scrobble data csv."""
        self._logger.info("Starting to link scrobbles...", separator=self._separator)
        self._scrobble_data_loader.load(sample_rows=TEST_SAMPLE_SIZE)

        # library_data: DataFrame = self._scrobble_data_loader.get_library_data()
        scrobble_data: DataFrame = self._scrobble_data_loader.get_scrobble_data()

        enriched_scrobble_data: DataFrame = self._scrobble_matcher.link_scrobbles(scrobble_data)
        self._log_unmatched_amount(enriched_scrobble_data)
        return enriched_scrobble_data

    def _log_unmatched_amount(self, enriched_scrobble_date: pd.DataFrame) -> None:
        # Log unmatched count
        unmatched: int = enriched_scrobble_date["track_uuid"].eq(NO_MATCH_LABEL).sum()
        total: int = len(enriched_scrobble_date)
        self._logger.info(
            f"Linked {total - unmatched} of {total} scrobbles. {unmatched} remain unmatched.",
            separator=self._separator
        )
