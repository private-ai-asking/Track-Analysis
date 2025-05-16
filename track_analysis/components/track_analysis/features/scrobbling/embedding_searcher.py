from typing import Tuple

import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.scrobbling.algorithm.algorithm_context import AlgorithmContext


class EmbeddingSearcher:
    """Utility class for searching embeddings."""
    def __init__(self,
                 logger: HoornLogger,
                 top_k: int):
        self._logger = logger
        self._separator: str = "EmbeddingSearcher"

        self._top_k = top_k

        # Pre-allocate FAISS output buffers
        n_queries = 10_000
        self._search_indices_buffer = np.empty((n_queries, self._top_k), dtype=np.int64)
        self._search_distances_buffer = np.empty((n_queries, self._top_k), dtype=np.float32)

        self._logger.trace("Initialized Successfully.", separator=self._separator)

    def search(self, embeddings: np.ndarray, ctx: AlgorithmContext) -> Tuple[np.array, np.array]:
        if embeddings.dtype != np.float32 or not embeddings.flags["C_CONTIGUOUS"]:
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        nq = embeddings.shape[0]
        Dret, Iret = ctx.library_index.search(embeddings, self._top_k)
        self._search_distances_buffer[:nq] = Dret
        self._search_indices_buffer[:nq] = Iret
        return self._search_indices_buffer[:nq], self._search_distances_buffer[:nq]
