import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from pandas import DataFrame
from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import StringUtils
from track_analysis.components.track_analysis.constants import CACHE_DIRECTORY, CLEAR_CACHE, TEST_SAMPLE_SIZE, \
    DATA_DIRECTORY
from track_analysis.components.track_analysis.features.scrobbling.scrobble_data_loader import ScrobbleDataLoader
from track_analysis.components.track_analysis.features.scrobbling.scrobble_matcher import ScrobbleMatcher


class ScrobbleLinkerService:
    """Used to link scrobbles with library-level data using fuzzy matching. High-level API."""

    def __init__(self,
                 logger: HoornLogger,
                 library_data_path: Path,
                 scrobble_data_path: Path,
                 string_utils: StringUtils,
                 minimum_fuzzy_threshold: float = 95.0):
        self._logger: HoornLogger = logger
        self._separator: str = "ScrobbleLinker"
        self._string_utils: StringUtils = string_utils
        self._combo_key: str = "||"
        self._embedder: SentenceTransformer = SentenceTransformer(model_name_or_path=str(DATA_DIRECTORY / "__internal__" / "all-MiniLM-l6-v2"), device="cuda")
        self._scrobble_data_loader: ScrobbleDataLoader = ScrobbleDataLoader(logger, library_data_path, scrobble_data_path, self._string_utils)

        cache_path: Path = CACHE_DIRECTORY.joinpath("scrobble_cache.json")
        keys_path: Path = DATA_DIRECTORY.joinpath("__internal__", "lib_keys.pkl")
        index_path: Path = DATA_DIRECTORY.joinpath("__internal__", "lib.index")

        if CLEAR_CACHE:
            cache_path.unlink(missing_ok=True)

        self._scrobble_matcher: ScrobbleMatcher = ScrobbleMatcher(
            logger,
            CacheBuilder(logger, cache_path, tree_separator=self._combo_key),
            embedder=self._embedder,
            threshold=minimum_fuzzy_threshold,
            key_combo=self._combo_key,
            ann_k=10,
            keys_path=keys_path,
            faiss_index_path=index_path
        )

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def build_embeddings_for_library(self) -> None:
        self._logger.info("Starting to build embeddings...", separator=self._separator)

        # 1) load a (small) sample to initialize loader, then pull full library
        self._scrobble_data_loader.load(sample_rows=TEST_SAMPLE_SIZE)
        library_data: DataFrame = self._scrobble_data_loader.get_library_data()

        # 2) build the combo strings
        combo = self._combo_key
        lib_strings = (
                library_data['_n_artist']
                + combo + library_data['_n_album']
                + combo + library_data['_n_title']
        ).tolist()

        # 3) embed each string via ONNXRuntime
        lib_embeddings = self._embedder.encode(
            lib_strings,
            batch_size=128,
            convert_to_numpy=True,
        ).astype('float32')

        # 4) persist raw embeddings and UUID list
        embeddings_path = DATA_DIRECTORY / "__internal__" / "lib_embeddings.npy"
        keys_path       = DATA_DIRECTORY / "__internal__" / "lib_keys.pkl"
        np.save(embeddings_path, lib_embeddings)

        with open(keys_path, 'wb') as f:
            pickle.dump(library_data['UUID'].tolist(), f)

        # 5) build Flat index & add vectors
        N, d = lib_embeddings.shape
        index = faiss.IndexFlatL2(d)
        index.add(lib_embeddings)

        # 6) write out index file
        index_path = DATA_DIRECTORY / "__internal__" / "lib.index"
        faiss.write_index(index, str(index_path))

        self._logger.info("Done building embeddings.", separator=self._separator)

    def _log_unmatched_amount(self, enriched_scrobble_date: pd.DataFrame) -> None:
        # Log unmatched count
        unmatched: int = enriched_scrobble_date["track_uuid"].eq("<NO ASSOCIATED KEY>").sum()
        total: int = len(enriched_scrobble_date)
        self._logger.info(
            f"Linked {total - unmatched} of {total} scrobbles. {unmatched} remain unmatched.",
            separator=self._separator
        )

    def link_scrobbles(self) -> pd.DataFrame:
        """Links scrobble data to library data by matching tracks and writing the associated Track ID
        (if any) into the enriched scrobble data csv."""
        self._logger.info("Starting to link scrobbles...", separator=self._separator)
        self._scrobble_data_loader.load(sample_rows=TEST_SAMPLE_SIZE)

        library_data: DataFrame = self._scrobble_data_loader.get_library_data()
        scrobble_data: DataFrame = self._scrobble_data_loader.get_scrobble_data()

        enriched_scrobble_data: DataFrame = self._scrobble_matcher.link_scrobbles(library_data, scrobble_data)
        self._log_unmatched_amount(enriched_scrobble_data)
        return enriched_scrobble_data
