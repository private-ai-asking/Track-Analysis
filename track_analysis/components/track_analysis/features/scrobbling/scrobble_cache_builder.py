import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import \
    SimilarityScorer
from track_analysis.components.track_analysis.features.scrobbling.algorithm.algorithm_context import AlgorithmContext
from track_analysis.components.track_analysis.features.scrobbling.algorithm.cache_builder_pipeline import \
    CacheBuilderPipeline
from track_analysis.components.track_analysis.features.scrobbling.model.scrabble_cache_algorithm_parameters import \
    ScrobbleCacheAlgorithmParameters
from track_analysis.components.track_analysis.features.scrobbling.scrobble_data_loader import ScrobbleDataLoader
from track_analysis.components.track_analysis.features.scrobbling.scrobble_utility import ScrobbleUtility


class ScrobbleCacheBuilder:
    """Builds and filters scrobbles for cache insertion.
    In test mode, thresholds are ignored and all candidates with valid UUIDs are accepted."""

    _SEPARATOR = 'ScrobbleCacheBuilder'

    def __init__(
            self,
            logger: HoornLogger,
            cache_builder: CacheBuilder,
            data_loader: ScrobbleDataLoader,
            scrobble_utils: ScrobbleUtility,
            index_path: Path,
            keys_path: Path,
            embedding_model: SentenceTransformer,
            parameters: ScrobbleCacheAlgorithmParameters = ScrobbleCacheAlgorithmParameters(),
            sample_size: Optional[int] = None,
            test: bool = False,
    ):
        self._logger = logger
        self._cache = cache_builder
        self._data_loader = data_loader
        self._index_path = index_path
        self._keys_path = keys_path

        self._sample_size = sample_size

        self._pipeline: CacheBuilderPipeline = CacheBuilderPipeline(
            logger=self._logger,
            scrobble_utils=scrobble_utils,
            embedder=embedding_model,
            parameters=parameters,
            test_mode=test
        )
        self._pipeline.build_pipeline()

        self._logger.debug("Initialized ScrobbleCacheBuilder.", separator=self._SEPARATOR)

    def build_cache(self) -> None:
        """Main entry point: loads data, applies exact matches and NN filtering, then saves cache."""
        self._logger.info("Starting cache build...", separator=self._SEPARATOR)
        library_lookup_key_to_uuid, library_keys, scrobble_data_frame, library_index, library_data_frame = self._prepare_data()

        initial_ctx: AlgorithmContext = AlgorithmContext(
            original_scrobble_count=len(scrobble_data_frame),
            previous_pipe_description="None",
            scrobble_data_frame=scrobble_data_frame,
            library_data_frame=library_data_frame,
            library_lookup_key_to_uuid=library_lookup_key_to_uuid,
            uncertain_keys=[],
            library_index=library_index,
            library_keys=library_keys
        )

        self._pipeline.flow(data=initial_ctx)

        self._cache.save()
        self._logger.info("Cache build complete.", separator=self._SEPARATOR)


    def _prepare_data(self) -> Tuple[Dict[str, str], List[str], pd.DataFrame, faiss.Index, pd.DataFrame]:
        """Load scrobble and library data, index, and lib_keys."""
        self._logger.debug("Loading data...", separator=self._SEPARATOR)
        self._data_loader.load(sample_rows=self._sample_size)

        library_lookup_key_to_uuid = self._data_loader.get_direct_lookup()
        library_data_frame = self._data_loader.get_library_data()
        scrobble_data_frame = self._data_loader.get_scrobble_data()
        library_index = faiss.read_index(str(self._index_path))

        with open(self._keys_path, "rb") as f:
            lib_keys = pickle.load(f)

        self._logger.debug("Data and keys loaded.", separator=self._SEPARATOR)

        return library_lookup_key_to_uuid, lib_keys, scrobble_data_frame, library_index, library_data_frame

