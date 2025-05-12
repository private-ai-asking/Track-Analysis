from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.md_common_python.py_common.utils import (
    gaussian_exponential_kernel_confidence_percentage,
    SimilarityScorer, CollectionExtensions,
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
        # Utils
        self._logger = logger
        self._scrobble_utils = scrobble_utils
        self._embedder = embedder
        self._scorer = scorer
        self._params = parameters
        self._test_mode = test_mode

        # Params
        self._accept_threshold = parameters.confidence_accept_threshold
        self._reject_threshold = parameters.confidence_reject_threshold
        self._token_threshold = parameters.token_accept_threshold / 100.0
        self._sigma = parameters.gaussian_sigma
        self._batch_size = parameters.batch_size
        self._top_k = parameters.top_k

        # Misc
        self._separator = "CacheBuilder.NearestNeighborSearch"

        self._accept_keys: List[str] = []
        self._reject_keys: List[str] = []
        self._uncertain_keys: List[str] = []

        self._logger.trace("Initialized NearestNeighbourSearch.", separator=self._separator)

    def flow(self, ctx: AlgorithmContext) -> AlgorithmContext:
        df_scrobbles = ctx.scrobble_data_frame
        if df_scrobbles.empty:
            self._log_empty()
            return ctx

        self._logger.debug(
            f"Starting NN phase for {len(df_scrobbles)} scrobbles.", separator=self._separator
        )

        embeddings = self._batch_encode(df_scrobbles['__key'].tolist())
        distances, indices = ctx.library_index.search(embeddings, self._top_k)

        for idx, key in enumerate(df_scrobbles['__key']):
            self._process_entry(idx, key, distances[idx][0], indices[idx][0], ctx)

        self._finalize(ctx)
        return ctx

    def _log_empty(self) -> None:
        self._logger.warning("No scrobbles to process.", separator=self._separator)

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
            distance, sigma=self._sigma
        )

        token_sim = self._compute_token_similarity(
            idx, uuid, ctx.scrobble_data_frame, ctx.library_data_frame
        )

        auto_accept, auto_reject, marked_uncertain = self._get_acceptance_state(confidence, token_sim)

        if self._test_mode or auto_accept:
            self._handle_accept(key, uuid, confidence, ctx.library_data_frame, auto_accept)
        if auto_reject:
            self._handle_rejection(key, confidence)
        if marked_uncertain:
            self._handle_uncertain(key, confidence, token_sim=token_sim)

    def _valid_index(self, index: int, size: int) -> bool:
        return 0 <= index < size

    def _get_acceptance_state(self, confidence: float, token_sim: Optional[float]) -> Tuple[bool, bool, bool]:
        is_confident, has_token_similarity_predicate = self._get_predicates(confidence, token_sim)

        # Conditions
        auto_accept = self._check_must_accept(is_confident, has_token_similarity_predicate)
        auto_reject = self._check_must_reject(is_confident, has_token_similarity_predicate, confidence)
        marked_uncertain = not auto_accept and not auto_reject

        self._check_for_multiple_states_warning(states=[auto_accept, auto_reject, marked_uncertain])

        return auto_accept, auto_reject, marked_uncertain

    @staticmethod
    def _check_must_accept(is_confident: bool, has_token_sim: bool) -> bool:
        return is_confident and has_token_sim

    def _check_must_reject(self, is_confident: bool, has_token_sim: bool, confidence: float):
        return (is_confident and not has_token_sim) or confidence <= self._reject_threshold

    def _get_predicates(self, confidence: float, token_sim: Optional[float]) -> Tuple[bool, bool]:
        is_confident = confidence >= self._accept_threshold
        has_token_similarity_predicate = (token_sim is None) or (token_sim >= self._token_threshold)
        return is_confident, has_token_similarity_predicate

    def _check_for_multiple_states_warning(self, states: List[bool]) -> None:
        if CollectionExtensions.more_than(lambda s: s, states, 1):
            self._logger.warning(f"The algorithm determined a match to be ({states[0]}-a/{states[1]}-r/{states[2]}-u); "
                                 f"only one of these can be true at the same time.", separator=self._separator)

    def _handle_accept(self, key: str, uuid: str, confidence: float, library_data_frame: pd.DataFrame, real_accept: bool) -> None:
        self._scrobble_utils.save_cache_item(
            key, uuid, confidence, library_data_frame
        )

        if real_accept:
            self._accept_keys.append(key)


    def _handle_rejection(self, key: str, confidence: float) -> None:
        if not self._test_mode:
            self._logger.debug(
                f"Auto-rejected {key} (confidence={confidence:.2f}).",
                separator=self._separator,
            )

        self._reject_keys.append(key)

    def _handle_uncertain(self, key: str, confidence: float, token_sim: Optional[float]) -> None:
        if not self._test_mode:
            reason = f"confidence={confidence:.2f}, token_sim={token_sim}"
            self._mark_uncertain(key, reason)
        else:
            self._uncertain_keys.append(key)

    def _mark_uncertain(self, key: str, reason: str) -> None:
        self._uncertain_keys.append(key)
        self._logger.debug(
            f"Marked uncertain: {key} ({reason})", separator=self._separator
        )

    def _compute_token_similarity(
            self,
            idx: int,
            uuid: str,
            scrobbles: pd.DataFrame,
            library: pd.DataFrame,
    ) -> float:
        scrobble = scrobbles.iloc[idx]
        lib_row = library[library['UUID'] == uuid].iloc[0]
        score = self._scorer.score(
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

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        self._logger.debug("Encoding batch.", separator=self._separator)
        batches = [
            texts[i : i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]
        embeddings_list = [
            self._embedder.encode(
                batch,
                convert_to_numpy=True,
                batch_size=len(batch),
                show_progress_bar=False,
            ).astype('float32')
            for batch in batches
        ]
        return np.vstack(embeddings_list)

    def _finalize(self, ctx: AlgorithmContext) -> None:
        self._logger.info(
            f"NN phase complete. Accepted: {len(self._accept_keys)}, "
            f"Rejected: {len(self._reject_keys)}, "
            f"Uncertain: {len(self._uncertain_keys)}",
            separator=self._separator,
        )

        # update context description
        ctx.previous_pipe_description = 'nearest neighbour search'

        # slice dataframes by keys
        df = ctx.scrobble_data_frame
        ctx.auto_accepted_scrobbles = df[df['__key'].isin(self._accept_keys)].copy()
        ctx.auto_rejected_scrobbles = df[df['__key'].isin(self._reject_keys)].copy()
        ctx.confused_scrobbles      = df[df['__key'].isin(self._uncertain_keys)].copy()

        # preserve uncertain keys list
        ctx.uncertain_keys = self._uncertain_keys

        # if test mode, clear main df but keep lists
        if self._test_mode:
            ctx.scrobble_data_frame = ctx.scrobble_data_frame.iloc[0:0]
