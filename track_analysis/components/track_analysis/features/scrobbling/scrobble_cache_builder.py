from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import gaussian_exponential_kernel_confidence_percentage
from track_analysis.components.track_analysis.features.scrobbling.scrobble_data_loader import ScrobbleDataLoader
from track_analysis.components.track_analysis.features.scrobbling.scrobble_utility import ScrobbleUtility


class ScrobbleCacheBuilder:
    """Builds and filters scrobbles for cache insertion."""

    _SEPARATOR = 'ScrobbleCacheBuilder'

    def __init__(
            self,
            logger: HoornLogger,
            cache_builder: CacheBuilder,
            data_loader: ScrobbleDataLoader,
            scrobble_utils: ScrobbleUtility,
            index_path: Path,
            embedding_model: SentenceTransformer,
            confidence_accept_threshold: float = 95.0,
            confidence_reject_threshold: float = 30.0,
            gaussian_sigma: float = 0.25,
            batch_size: int = 64,
            top_k: int = 2,
            sample_size: int = None,
    ):
        self._logger = logger
        self._cache = cache_builder
        self._data_loader = data_loader
        self._scrobble_utils = scrobble_utils
        self._index_path = index_path
        self._embedder = embedding_model

        self._c_accept = confidence_accept_threshold
        self._c_reject = confidence_reject_threshold
        self._sigma = gaussian_sigma
        self._batch_size = batch_size
        self._top_k = top_k
        self._sample_size = sample_size

        self._uncertain_keys: List[str] = []
        self._logger.debug("Initialized ScrobbleCacheBuilder.", separator=self._SEPARATOR)

    def build_cache(self) -> None:
        self._logger.info("Starting cache build...", separator=self._SEPARATOR)
        lookup, lib_keys, scrobbles, index = self._prepare_data()
        total = len(scrobbles)

        uniques = self._extract_unique_keys(scrobbles)
        self._log_status(len(uniques), total, "unique entries after flattening.")

        to_process = self._apply_exact_matches(uniques, lookup)
        self._log_status(len(to_process), total, "unique entries after exact-match filtering.")

        remaining = self._apply_nn_filter(to_process, index, lookup, lib_keys)
        self._log_status(len(remaining), total, "unique entries after NN confidence filtering.")

        self._cache.save()
        self._logger.info("Cache build complete.", separator=self._SEPARATOR)

        self.print_uncertain()

    def print_uncertain(self) -> None:
        """Logs keys that require manual review."""
        if not self._uncertain_keys:
            self._logger.info("No uncertain entries.", separator=self._SEPARATOR)
        else:
            self._logger.info("Uncertain keys for manual review:", separator=self._SEPARATOR)
            for key in self._uncertain_keys:
                self._logger.info(f"  {key}", separator=self._SEPARATOR)

    def _log_status(self, current: int, total: int, message: str) -> None:
        percent = (current / total * 100) if total else 0
        self._logger.info(f"({current}/{total}) [{percent:.2f}%] {message}", separator=self._SEPARATOR)

    def _prepare_data(self) -> Tuple[Dict[str, str], List[str], pd.DataFrame, faiss.Index]:
        self._logger.debug("Loading data...", separator=self._SEPARATOR)
        self._data_loader.load(sample_rows=self._sample_size)
        lookup = self._data_loader.get_direct_lookup()
        df = self._data_loader.get_scrobble_data()
        index = faiss.read_index(str(self._index_path))
        self._logger.debug("Data loaded and index read.", separator=self._SEPARATOR)
        return lookup, list(lookup.keys()), df, index

    def _extract_unique_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        self._logger.debug("Extracting unique keys.", separator=self._SEPARATOR)
        df = df.copy()
        df['__key'] = df.apply(
            lambda r: self._scrobble_utils.compute_key(r['_n_title'], r['_n_artist'], r['_n_album']),
            axis=1,
        )
        return df.drop_duplicates('__key').reset_index(drop=True)

    def _apply_exact_matches(self, df: pd.DataFrame, lookup: Dict[str, str]) -> pd.DataFrame:
        self._logger.debug("Applying exact match filter.", separator=self._SEPARATOR)
        df = df.copy()
        df['uuid'] = df['__key'].map(lookup)
        matches = df[df['uuid'].notna()]
        for key, uuid in zip(matches['__key'], matches['uuid']):
            self._cache.set(key, uuid)
            self._logger.debug(f"Exact match: {key} -> {uuid}", separator=self._SEPARATOR)
        return df[df['uuid'].isna()].reset_index(drop=True)

    def _apply_nn_filter(
            self,
            df: pd.DataFrame,
            index: faiss.Index,
            lookup: Dict[str, str],
            lib_keys: List[str],
    ) -> pd.DataFrame:
        if df.empty:
            self._logger.debug("No data for NN filtering.", separator=self._SEPARATOR)
            return df

        self._logger.debug(f"Applying NN filter to {len(df)} entries.", separator=self._SEPARATOR)
        keys = df['__key'].tolist()
        embs = self._batch_encode(keys)
        distances, indices = index.search(embs, self._top_k)

        keep_idxs: List[int] = []
        for i, key in enumerate(keys):
            nn_idx = indices[i][0]
            if nn_idx < 0 or nn_idx >= len(lib_keys):
                self._logger.debug(f"Invalid NN index for {key}: {nn_idx}", separator=self._SEPARATOR)
                self._uncertain_keys.append(key)
                keep_idxs.append(i)
                continue

            dist = distances[i][0]
            nn_key = lib_keys[nn_idx]
            cfp = gaussian_exponential_kernel_confidence_percentage(dist, sigma=self._sigma)

            if cfp >= self._c_accept and (uuid := lookup.get(nn_key)):
                self._cache.set(key, uuid)
                self._logger.debug(f"Auto-accepted: {key} -> {uuid} (cfp={cfp:.2f})", separator=self._SEPARATOR)
            elif cfp <= self._c_reject:
                self._logger.debug(f"Auto-rejected: {key} (cfp={cfp:.2f})", separator=self._SEPARATOR)
            else:
                keep_idxs.append(i)
                self._uncertain_keys.append(key)
                self._logger.debug(f"Marked uncertain: {key} (cfp={cfp:.2f})", separator=self._SEPARATOR)

        self._logger.info(f"Filtering complete. Uncertain: {len(self._uncertain_keys)}", separator=self._SEPARATOR)
        return df.iloc[keep_idxs].reset_index(drop=True)

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        self._logger.debug("Encoding batch of keys.", separator=self._SEPARATOR)
        batches = [texts[i : i + self._batch_size] for i in range(0, len(texts), self._batch_size)]
        embeddings: List[np.ndarray] = []
        for batch in batches:
            embeddings.append(
                self._embedder.encode(
                    batch,
                    convert_to_numpy=True,
                    batch_size=len(batch),
                    show_progress_bar=False,
                ).astype('float32')
            )
        return np.vstack(embeddings)
