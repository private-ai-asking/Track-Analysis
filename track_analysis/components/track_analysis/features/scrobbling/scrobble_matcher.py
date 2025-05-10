import pickle
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import gaussian_exponential_kernel_confidence_percentage
from track_analysis.components.track_analysis.constants import CACHE_DIRECTORY, NO_MATCH_LABEL, CLEAR_CACHE

# Path for persistent embedding cache
_EMBED_CACHE_PATH = Path(CACHE_DIRECTORY) / "scrobble_embed_cache.pkl"

if CLEAR_CACHE:
    _EMBED_CACHE_PATH.unlink(missing_ok=True)


def load_embed_cache() -> dict:
    """
    Load the on-disk embedding cache, or return empty dict if none.
    """
    try:
        with open(_EMBED_CACHE_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def save_embed_cache(cache: dict) -> None:
    """
    Persist embedding cache to disk.
    """
    _EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_EMBED_CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)


class ScrobbleMatcher:
    """
    Matches scrobbles to library tracks via FAISS ANN and confidence thresholding.
    """

    def __init__(
            self,
            logger: HoornLogger,
            cache_builder: CacheBuilder,
            embedder: SentenceTransformer,
            faiss_index_path: Path,
            keys_path: Path,
            combo_key: str = "||",
            threshold: float = 90.0,
            sigma: float = 0.35,
            ann_k: int = 5,
            batch_size: int = 64
    ):
        self._logger = logger
        self._cache = cache_builder
        self._embedder = embedder
        self._threshold = threshold
        self._sigma = sigma
        self._ann_k = ann_k
        self._batch_size = batch_size
        self._combo_key = combo_key

        # Load FAISS index and UUIDs
        self._index, self._uuids = self._load_faiss(faiss_index_path, keys_path)

        # In-memory embedding cache
        self._embed_cache = load_embed_cache()

    def _load_faiss(self, index_path: Path, keys_path: Path):
        self._logger.info(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(str(index_path))
        with open(keys_path, 'rb') as f:
            uuids = pickle.load(f)
        assert index.ntotal == len(uuids), "Index size mismatch"
        return index, uuids

    def _create_key(self, artist: str, album: str, title: str) -> str:
        """
        Create a composite key from artist, album, and title.
        """
        return f"{artist}{self._combo_key}{album}{self._combo_key}{title}"

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Compute or retrieve cached embeddings for given texts.
        """
        unseen = [t for t in texts if t not in self._embed_cache]
        if unseen:
            embs = self._embedder.encode(
                unseen,
                convert_to_numpy=True,
                batch_size=self._batch_size,
                show_progress_bar=False
            ).astype('float32')
            for t, e in zip(unseen, embs):
                self._embed_cache[t] = e
        return np.vstack([self._embed_cache[t] for t in texts])

    def _batch_search(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run FAISS search on batch of embeddings.
        Returns distances and indices arrays.
        """
        return self._index.search(embeddings, self._ann_k)

    def _process_record(self, text: str, dist_row: np.ndarray, idx_row: np.ndarray) -> tuple[str, float]:
        """
        Determine best match and confidence for a single record.
        Returns (match_uuid_or_label, confidence_percent).
        """
        best_pos = int(np.argmin(dist_row))
        best_dist = float(dist_row[best_pos])
        best_uuid = self._uuids[idx_row[best_pos]]

        # Compute confidence in [0,100]
        conf_pct = gaussian_exponential_kernel_confidence_percentage(best_dist, sigma=self._sigma)

        # Decide match based on threshold
        if conf_pct >= self._threshold:
            return best_uuid, conf_pct
        return NO_MATCH_LABEL, conf_pct

    def _initialize_results(self, texts: list[str]) -> tuple[list, list]:
        """
        Initialize results list and indices to process based on cache hits.
        """
        results = []  # (uuid, conf)
        to_process = []
        for i, txt in enumerate(texts):
            cached = self._cache.get(txt)
            if cached is not None:
                results.append((cached, None))
            else:
                results.append((None, None))
                to_process.append(i)
        return results, to_process

    def link_scrobbles(
            self,
            scrobble_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        For each scrobble record, find best match and attach track_uuid and confidence.
        Unmatched items get NO_MATCH_LABEL.
        """
        records = scrobble_df.to_dict(orient='records')
        texts = [self._create_key(r['_n_artist'], r['_n_album'], r['_n_title']) for r in records]

        # Initialize results and determine which to process
        results, to_process = self._initialize_results(texts)

        if to_process:
            embeddings = self._get_embeddings(texts)
            dists, idxs = self._batch_search(embeddings)

            for idx in to_process:
                match, conf = self._process_record(texts[idx], dists[idx], idxs[idx])
                results[idx] = (match, conf)
                self._cache.set(texts[idx], match)

        # Attach results to DataFrame
        uuids = [m or NO_MATCH_LABEL for m, _ in results]
        confidences = [c for _, c in results]
        output = scrobble_df.copy()
        output['track_uuid'] = uuids
        output['confidence'] = confidences

        # Persist caches
        self._cache.save()
        save_embed_cache(self._embed_cache)

        return output
