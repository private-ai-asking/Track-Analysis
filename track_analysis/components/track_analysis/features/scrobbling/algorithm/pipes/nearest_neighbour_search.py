from typing import Optional, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.md_common_python.py_common.utils import (
    gaussian_exponential_kernel_confidence_percentage,
    SimilarityScorer,
)
from track_analysis.components.track_analysis.features.scrobbling.algorithm.algorithm_context import AlgorithmContext
from track_analysis.components.track_analysis.features.scrobbling.model.scrabble_cache_algorithm_parameters import (
    ScrobbleCacheAlgorithmParameters,
)
from track_analysis.components.track_analysis.features.scrobbling.scrobble_utility import ScrobbleUtility


class NearestNeighbourSearch(IPipe):
    """Encapsulates nearest neighbour filtering for scrobble analysis."""

    def __init__(
            self,
            logger: HoornLogger,
            scrobble_utils: ScrobbleUtility,
            embedder: SentenceTransformer,
            scorer: SimilarityScorer,
            parameters: ScrobbleCacheAlgorithmParameters,
            test_mode: bool,
    ):
        self.logger = logger
        self.scrobble_utils = scrobble_utils
        self.embedder = embedder
        self.scorer = scorer
        self.params = parameters
        self.test_mode = test_mode

        self.accept_threshold = parameters.confidence_accept_threshold
        self.reject_threshold = parameters.confidence_reject_threshold
        self.token_threshold = parameters.token_accept_threshold / 100.0
        self.sigma = parameters.gaussian_sigma
        self.batch_size = parameters.batch_size
        self.top_k = parameters.top_k

        self.separator = "CacheBuilder.NearestNeighborSearch"
        self.uncertain_keys: List[str] = []

        self.logger.trace("Initialized NearestNeighbourSearch.", separator=self.separator)

    def flow(self, ctx: AlgorithmContext) -> AlgorithmContext:
        df_scrobbles = ctx.scrobble_data_frame
        if df_scrobbles.empty:
            self._log_empty()
            return ctx

        self.logger.debug(
            f"Starting NN phase for {len(df_scrobbles)} scrobbles.", separator=self.separator
        )

        embeddings = self._batch_encode(df_scrobbles['__key'].tolist())
        distances, indices = ctx.library_index.search(embeddings, self.top_k)

        for idx, key in enumerate(df_scrobbles['__key']):
            self._process_entry(idx, key, distances[idx][0], indices[idx][0], ctx)

        self._finalize(ctx)
        return ctx

    def _log_empty(self) -> None:
        self.logger.warning("No scrobbles to process.", separator=self.separator)

    def _process_entry(
            self,
            idx: int,
            key: str,
            distance: float,
            lib_index: int,
            ctx: AlgorithmContext,
    ) -> None:
        if not self._valid_index(lib_index, len(ctx.library_keys)):
            self._mark_uncertain(key, 'invalid index')
            return

        uuid = ctx.library_keys[lib_index]
        confidence = gaussian_exponential_kernel_confidence_percentage(
            distance, sigma=self.sigma
        )
        token_sim = None
        if not self.test_mode:
            token_sim = self._compute_token_similarity(
                idx, uuid, ctx.scrobble_data_frame, ctx.library_data_frame
            )

        if self.test_mode or self._accept(confidence, token_sim):
            self.scrobble_utils.save_cache_item(
                key, uuid, confidence, ctx.library_data_frame
            )
        else:
            self._handle_rejection(key, confidence, token_sim)

    def _valid_index(self, index: int, size: int) -> bool:
        return 0 <= index < size

    def _accept(self, confidence: float, token_sim: Optional[float]) -> bool:
        is_confident = confidence >= self.accept_threshold
        has_tokens = token_sim is None or token_sim >= self.token_threshold
        return is_confident and has_tokens

    def _handle_rejection(
            self, key: str, confidence: float, token_sim: Optional[float]
    ) -> None:
        if confidence <= self.reject_threshold:
            self.logger.debug(
                f"Auto-rejected {key} (confidence={confidence:.2f}).",
                separator=self.separator,
            )
        else:
            reason = f"confidence={confidence:.2f}, token_sim={token_sim}"
            self._mark_uncertain(key, reason)

    def _compute_token_similarity(
            self,
            idx: int,
            uuid: str,
            scrobbles: pd.DataFrame,
            library: pd.DataFrame,
    ) -> float:
        scrobble = scrobbles.iloc[idx]
        lib_row = library[library['UUID'] == uuid].iloc[0]
        score = self.scorer.score(
            rec1={
                "title": scrobble['_n_title'],
                "artist": scrobble['_n_artist'],
                "album": scrobble['_n_album'],
            },
            rec2={
                "title": lib_row['_n_title'],
                "artist": lib_row['_n_artist'],
                "album": lib_row['_n_album'],
            },
            optimize=True,
        )
        return score / 100.0

    def _mark_uncertain(self, key: str, reason: str) -> None:
        self.uncertain_keys.append(key)
        self.logger.debug(
            f"Marked uncertain: {key} ({reason})", separator=self.separator
        )

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        self.logger.debug("Encoding batch.", separator=self.separator)
        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        embeddings_list = [
            self.embedder.encode(
                batch,
                convert_to_numpy=True,
                batch_size=len(batch),
                show_progress_bar=False,
            ).astype('float32')
            for batch in batches
        ]
        return np.vstack(embeddings_list)

    def _finalize(self, ctx: AlgorithmContext) -> None:
        self.logger.info(
            f"NN phase complete. Uncertain count: {len(self.uncertain_keys)}",
            separator=self.separator,
        )
        ctx.previous_pipe_description = 'nearest neighbour search'
        ctx.uncertain_keys = self.uncertain_keys
        if self.test_mode:
            ctx.scrobble_data_frame = ctx.scrobble_data_frame.iloc[0:0]
