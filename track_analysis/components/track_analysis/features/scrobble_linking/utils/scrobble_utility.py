from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.logging import HoornLogger


class ScrobbleUtility:
    """Utility class for helpful misc methods relating to scrobble analysis."""
    def __init__(self,
                 logger: HoornLogger,
                 embedder: SentenceTransformer,
                 embed_weights: Dict,
                 embed_batch_size: int = 64,
                 join_key: str = "||"):
        self._logger = logger
        self._separator = "ScrobbleUtility"

        self._embedder = embedder
        self._weights = embed_weights

        self._join_key: str = join_key

        self._batch_size = embed_batch_size

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def compute_key(self, normalized_title: str, normalized_artists: str, normalized_album: str) -> str:
        combo_key: str = self._join_key
        return f"{normalized_artists}{combo_key}{normalized_album}{combo_key}{normalized_title}"



    def build_combined_embeddings(self,
                                  titles: List[str],
                                  artists: List[str],
                                  albums: List[str]) -> np.ndarray:
        """
        Single-pass encode → normalize → weight → concat.
        """
        n = len(titles)
        # 1) one big encode
        all_texts = titles + artists + albums
        # tell your embedder to give you float32 directly if possible
        embs = self._embedder.encode(
            all_texts,
            convert_to_numpy=True,
            dtype=np.float32,       # if your API supports it
            batch_size=self._batch_size,
            device='cuda'
        )  # shape: (3*n, D)

        # 2) split back into (n, D) chunks
        D = embs.shape[1]
        embs = embs.reshape(3, n, D)
        # embs[0] = titles, embs[1] = artists, embs[2] = albums

        # 3) L2‐normalize each “slice” [vectorized]
        #    normed = embs / sqrt(sum(embs**2, axis=2, keepdims=True))
        norms = np.linalg.norm(embs, axis=2, keepdims=True)
        normed = embs / (norms + 1e-12)  # avoid div-by-zero

        # 4) apply weights [broadcast over (3, n, D)]
        w = np.array([self._weights['title'],
                      self._weights['artist'],
                      self._weights['album']], dtype=np.float32)
        weighted = normed * w[:, None, None]

        # 5) build output: concatenate along D → shape (n, 3*D)
        combined = np.empty((n, 3*D), dtype=np.float32)
        # fill slices
        combined[:,        :D] = weighted[0]
        combined[:,    D:2*D] = weighted[1]
        combined[:, 2*D:3*D] = weighted[2]

        return combined

