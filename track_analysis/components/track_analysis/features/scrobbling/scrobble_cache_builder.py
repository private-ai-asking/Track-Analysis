import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import \
    gaussian_exponential_kernel_confidence_percentage, SimilarityScorer
from track_analysis.components.track_analysis.features.scrobbling.model.scrabble_cache_algorithm_parameters import \
    ScrobbleCacheAlgorithmParameters
from track_analysis.components.track_analysis.features.scrobbling.model.scrobble_cache_model import \
    ScrobbleCacheItemModel
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
            paramaters: ScrobbleCacheAlgorithmParameters = ScrobbleCacheAlgorithmParameters(),
            sample_size: Optional[int] = None,
            test: bool = False,
    ):
        self._logger = logger
        self._cache = cache_builder
        self._data_loader = data_loader
        self._scrobble_utils = scrobble_utils
        self._index_path = index_path
        self._keys_path = keys_path
        self._embedder = embedding_model
        self._scorer = SimilarityScorer(
            logger=logger,
            threshold=paramaters.token_accept_threshold,
            field_weights={}
        )

        self._c_accept = paramaters.confidence_accept_threshold
        self._c_reject = paramaters.confidence_reject_threshold
        self._t_accept = paramaters.token_accept_threshold / 100.0
        self._sigma = paramaters.gaussian_sigma
        self._batch_size = paramaters.batch_size
        self._top_k = paramaters.top_k
        self._sample_size = sample_size
        self._test = test

        self._uncertain_keys: List[str] = []
        self._logger.debug("Initialized ScrobbleCacheBuilder.", separator=self._SEPARATOR)

    def build_cache(self) -> None:
        """Main entry point: loads data, applies exact matches and NN filtering, then saves cache."""
        self._logger.info("Starting cache build...", separator=self._SEPARATOR)
        lookup, lib_keys, scrobbles, index, lib_df = self._prepare_data()

        uniques = self._extract_unique_keys(scrobbles)
        self._log_progress(len(uniques), len(scrobbles), "unique entries extracted.")

        remaining = self._apply_exact_match(uniques, lookup, lib_df)
        self._log_progress(len(remaining), len(scrobbles), "after exact-match filtering.")

        self._apply_nn_phase(remaining, lib_keys, lib_df, index)
        self._log_progress(len(self._uncertain_keys) if not self._test else 0,
                           len(scrobbles),
                           "NN phase complete; uncertainties logged.")

        self._cache.save()
        self._logger.info("Cache build complete.", separator=self._SEPARATOR)
        self._report_uncertain()

    def _apply_nn_phase(
            self,
            df: pd.DataFrame,
            lib_keys: List[str],
            lib_df: pd.DataFrame,
            index: faiss.Index,
    ) -> None:
        """Apply nearest-neighbor filtering, accepting everything under test mode or by thresholds otherwise."""
        if df.empty:
            return
        self._logger.debug(f"Applying NN phase to {len(df)} entries.", separator=self._SEPARATOR)

        keys = df['__key'].tolist()
        embeddings = self._batch_encode(keys)
        distances, indices = index.search(embeddings, self._top_k)

        for i, key in enumerate(keys):
            nn_idx = indices[i][0]
            if not (0 <= nn_idx < len(lib_keys)):
                self._mark_uncertain(key, "invalid NN index")
                continue

            uuid = lib_keys[nn_idx]
            cfp = gaussian_exponential_kernel_confidence_percentage(
                distances[i][0], sigma=self._sigma
            )
            token_sim = None if self._test else self._compute_token_similarity(i, uuid, lib_df)

            if self._test or self._evaluate_thresholds(cfp, token_sim):
                self._save_cache_item(key, uuid, cfp, lib_df)
            elif cfp <= self._c_reject:
                self._logger.debug(f"Auto-rejected: {key} (cfp={cfp:.2f})", separator=self._SEPARATOR)
            else:
                self._mark_uncertain(key, f"cfp={cfp:.2f}, token_sim={token_sim}")

        self._logger.info(f"NN phase complete. Uncertain: {len(self._uncertain_keys)}", separator=self._SEPARATOR)

    def _evaluate_thresholds(self, cfp: float, token_sim: Optional[float]) -> bool:
        return cfp >= self._c_accept and (token_sim is None or token_sim >= self._t_accept)

    def _apply_exact_match(
            self,
            df: pd.DataFrame,
            lookup: Dict[str, str],
            lib_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Map any scrobble-key exact matches to cache with full confidence."""
        self._logger.debug("Applying exact match.", separator=self._SEPARATOR)
        df = df.copy()
        df['uuid'] = df['__key'].map(lookup)
        for _, row in df[df['uuid'].notna()].iterrows():
            self._save_cache_item(row['__key'], row['uuid'], 100.0, lib_df)
        return df[df['uuid'].isna()].drop(columns=['uuid']).reset_index(drop=True)

    def _compute_token_similarity(
            self,
            idx: int,
            uuid: str,
            lib_df: pd.DataFrame
    ) -> float:
        """Compute token-level similarity between scrobble idx and library row for uuid."""
        scrobble = self._data_loader.get_scrobble_data().iloc[idx]
        lib_row = lib_df[lib_df['UUID'] == uuid].iloc[0]
        return self._scorer.score(
            rec1={"title": scrobble['_n_title'], "artist": scrobble['_n_artist'], "album": scrobble['_n_album']},
            rec2={"title": lib_row['_n_title'], "artist": lib_row['_n_artist'], "album": lib_row['_n_album']},
            optimize=True
        ) / 100.0

    def _save_cache_item(
            self,
            key: str,
            uuid: str,
            confidence: float,
            lib_df: pd.DataFrame
    ) -> None:
        """Instantiate and store a ScrobbleCacheItemModel in cache."""
        lib_row = lib_df[lib_df['UUID'] == uuid].iloc[0]
        item = ScrobbleCacheItemModel(
            associated_uuid=uuid,
            associated_track_title=lib_row['Title'],
            associated_track_album=lib_row['Album'],
            associated_track_artist=lib_row['Artist(s)'],
            confidence_factor_percentage=confidence,
        )
        self._cache.set(key, item.model_dump(mode='json'))
        self._logger.debug(f"Accepted: {key} -> {uuid} (confidence={confidence:.2f})", separator=self._SEPARATOR)

    def _mark_uncertain(
            self,
            key: str,
            reason: str
    ) -> None:
        """Mark a key as uncertain for manual review."""
        self._uncertain_keys.append(key)
        self._logger.debug(f"Marked uncertain: {key} ({reason})", separator=self._SEPARATOR)

    def _extract_unique_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute a unique key for each scrobble and drop duplicates."""
        self._logger.debug("Extracting unique keys.", separator=self._SEPARATOR)
        df = df.copy()
        df['__key'] = df.apply(
            lambda r: self._scrobble_utils.compute_key(r['_n_title'], r['_n_artist'], r['_n_album']),
            axis=1
        )
        return df.drop_duplicates('__key').reset_index(drop=True)

    def _prepare_data(self) -> Tuple[Dict[str, str], List[str], pd.DataFrame, faiss.Index, pd.DataFrame]:
        """Load scrobble and library data, index, and lib_keys."""
        self._logger.debug("Loading data...", separator=self._SEPARATOR)
        self._data_loader.load(sample_rows=self._sample_size)
        lookup = self._data_loader.get_direct_lookup()
        lib_df = self._data_loader.get_library_data()
        scrobble_df = self._data_loader.get_scrobble_data()
        index = faiss.read_index(str(self._index_path))
        with open(self._keys_path, "rb") as f:
            lib_keys = pickle.load(f)
        self._logger.debug("Data and keys loaded.", separator=self._SEPARATOR)
        return lookup, lib_keys, scrobble_df, index, lib_df

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts in batches."""
        self._logger.debug("Encoding batch.", separator=self._SEPARATOR)
        batches = [texts[i:i + self._batch_size] for i in range(0, len(texts), self._batch_size)]
        embeddings = [
            self._embedder.encode(
                batch,
                convert_to_numpy=True,
                batch_size=len(batch),
                show_progress_bar=False
            ).astype('float32')
            for batch in batches
        ]
        return np.vstack(embeddings)

    def _log_progress(self, count: int, total: int, message: str) -> None:
        pct = (count / total * 100) if total else 0
        self._logger.info(f"({count}/{total}) [{pct:.2f}%] {message}", separator=self._SEPARATOR)

    def _report_uncertain(self) -> None:
        """Log any keys marked uncertain."""
        if not self._uncertain_keys:
            self._logger.info("No uncertain entries.", separator=self._SEPARATOR)
        else:
            self._logger.info("Uncertain keys for review:", separator=self._SEPARATOR)
            for key in self._uncertain_keys:
                self._logger.info(f"  {key}", separator=self._SEPARATOR)
