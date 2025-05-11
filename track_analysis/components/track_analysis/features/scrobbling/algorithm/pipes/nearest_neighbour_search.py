from typing import Optional, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.md_common_python.py_common.utils import \
    gaussian_exponential_kernel_confidence_percentage, SimilarityScorer
from track_analysis.components.track_analysis.features.scrobbling.algorithm.algorithm_context import AlgorithmContext
from track_analysis.components.track_analysis.features.scrobbling.model.scrabble_cache_algorithm_parameters import \
    ScrobbleCacheAlgorithmParameters
from track_analysis.components.track_analysis.features.scrobbling.scrobble_utility import ScrobbleUtility


class NearestNeighbourSearch(IPipe):
    """Class used to encapsulate the nearest neighbor search functionality within the context of scrobble analysis."""
    def __init__(self,
                 logger: HoornLogger,
                 scrobble_utils: ScrobbleUtility,
                 embedder: SentenceTransformer,
                 scorer: SimilarityScorer,
                 parameters: ScrobbleCacheAlgorithmParameters,
                 test_mode: bool):
        self._logger: HoornLogger = logger
        self._separator: str = "CacheBuilder.NearestNeighborSearch"

        # Utils
        self._scrobble_utils: ScrobbleUtility = scrobble_utils
        self._embedder: SentenceTransformer = embedder
        self._scorer: SimilarityScorer = scorer

        # Flags
        self._test=test_mode

        # Params
        self._c_accept = parameters.confidence_accept_threshold
        self._c_reject = parameters.confidence_reject_threshold
        self._t_accept = parameters.token_accept_threshold / 100.0
        self._sigma = parameters.gaussian_sigma
        self._batch_size = parameters.batch_size
        self._top_k = parameters.top_k

        # On-Execution
        self._uncertain_keys: List[str] = []

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def flow(self, ctx: AlgorithmContext) -> AlgorithmContext:
        """Apply nearest-neighbor filtering, accepting everything under test mode or by thresholds otherwise."""
        scrobble_data_frame: pd.DataFrame = ctx.scrobble_data_frame
        library_data_frame: pd.DataFrame = ctx.library_data_frame

        if scrobble_data_frame.empty:
            self._logger.warning("There are no scrobbles to match. Skipping.", separator=self._separator)
            return ctx

        self._logger.debug(f"Applying NN phase to {len(scrobble_data_frame)} entries.", separator=self._separator)

        keys = scrobble_data_frame['__key'].tolist()
        embeddings = self._batch_encode(keys)
        distances, indices = ctx.library_index.search(embeddings, self._top_k)

        for i, key in enumerate(keys):
            nn_idx = indices[i][0]
            if not (0 <= nn_idx < len(ctx.library_keys)):
                self._mark_uncertain(key, "invalid NN index")
                continue

            uuid = ctx.library_keys[nn_idx]
            cfp = gaussian_exponential_kernel_confidence_percentage(
                distances[i][0], sigma=self._sigma
            )
            token_sim = None if self._test else self._compute_token_similarity(i, uuid, library_data_frame)

            if self._test or self._evaluate_thresholds(cfp, token_sim):
                self._scrobble_utils.save_cache_item(key, uuid, cfp, library_data_frame)
            elif cfp <= self._c_reject:
                self._logger.debug(f"Auto-rejected: {key} (cfp={cfp:.2f})", separator=self._separator)
            else:
                self._mark_uncertain(key, f"cfp={cfp:.2f}, token_sim={token_sim}")

        self._logger.info(f"NN phase complete. Uncertain: {len(self._uncertain_keys)}", separator=self._separator)
        ctx.previous_pipe_description = "performing nearest neighbor search"
        ctx.uncertain_keys = self._uncertain_keys

        if self._test:
            ctx.scrobble_data_frame = ctx.scrobble_data_frame.loc[[]]

        return ctx

    def _evaluate_thresholds(self, cfp: float, token_sim: Optional[float]) -> bool:
        return cfp >= self._c_accept and (token_sim is None or token_sim >= self._t_accept)


    def _compute_token_similarity(self, idx: int, uuid: str, scrobble_data_frame: pd.DataFrame, library_data_frame: pd.DataFrame) -> float:
        """Compute token-level similarity between scrobble idx and library row for uuid."""
        scrobble = scrobble_data_frame.iloc[idx]
        lib_row = library_data_frame[library_data_frame['UUID'] == uuid].iloc[0]
        return self._scorer.score(
            rec1={"title": scrobble['_n_title'], "artist": scrobble['_n_artist'], "album": scrobble['_n_album']},
            rec2={"title": lib_row['_n_title'], "artist": lib_row['_n_artist'], "album": lib_row['_n_album']},
            optimize=True
        ) / 100.0

    def _mark_uncertain(self, key: str, reason: str) -> None:
        """Mark a key as uncertain for manual review."""
        self._uncertain_keys.append(key)
        self._logger.debug(f"Marked uncertain: {key} ({reason})", separator=self._separator)

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts in batches."""
        self._logger.debug("Encoding batch.", separator=self._separator)
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
