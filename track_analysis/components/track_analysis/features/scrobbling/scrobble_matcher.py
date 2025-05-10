from pathlib import Path
import pickle
from typing import List, Optional, Dict, Tuple

import numpy as np
import faiss
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import pandas as pd

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import SimilarityScorer
from track_analysis.components.track_analysis.constants import DATA_DIRECTORY

# On-disk cache for scrobble embeddings
_EMBED_CACHE_PATH = Path(DATA_DIRECTORY) / "__internal__" / "scrobble_embed_cache.pkl"


def load_embed_cache() -> Dict[str, np.ndarray]:
    try:
        with open(_EMBED_CACHE_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def save_embed_cache(cache: Dict[str, np.ndarray]) -> None:
    _EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_EMBED_CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)


class ScrobbleMatcher:
    """Match scrobbles to library tracks using FAISS ANN + cached embeddings + batched fuzzy rerank."""

    _SEPARATOR = "ScrobbleMatcher"

    def __init__(
            self,
            logger: HoornLogger,
            cache_builder: CacheBuilder,
            embedder: SentenceTransformer,
            faiss_index_path: Path,
            keys_path: Path,
            key_combo: str = "||",
            field_weights: Optional[Dict[str, float]] = None,
            similarity_func= fuzz.token_sort_ratio,
            threshold: float = 95.0,
            ann_k: int = 5,
            batch_size: int = 64,
    ):
        self._logger = logger
        self._cache = cache_builder
        self._embedder = embedder
        self._key_combo = key_combo
        self._threshold = threshold
        self._ann_k = ann_k
        self._batch_size = batch_size

        self._load_faiss(faiss_index_path, keys_path)
        self._init_scorer(field_weights or {}, similarity_func)

    def _load_faiss(self, index_path: Path, keys_path: Path) -> None:
        self._logger.info("Loading FAISS index and UUIDs...", separator=self._SEPARATOR)
        self._index = faiss.read_index(str(index_path))
        with open(keys_path, 'rb') as f:
            self._uuids: List[str] = pickle.load(f)
        assert self._index.ntotal == len(self._uuids), "Index size and UUID list length mismatch"
        if hasattr(self._index, 'is_trained'):
            assert self._index.is_trained, "FAISS index is not trained"
        self._logger.trace("FAISS index ready.", separator=self._SEPARATOR)

    def _init_scorer(self, field_weights: Dict[str, float], similarity_func: callable) -> None:
        weights = field_weights or {'_n_title': 0.5, '_n_artist': 0.3, '_n_album': 0.2}
        self._scorer = SimilarityScorer(weights, self._logger, similarity_func, self._threshold)
        self._logger.trace("SimilarityScorer initialized.", separator=self._SEPARATOR)

    def _make_cache_key(self, rec: Dict) -> str:
        return f"{rec['_n_artist']}{self._key_combo}{rec['_n_album']}{self._key_combo}{rec['_n_title']}"

    def _prepare_library(self, library_df: pd.DataFrame) -> Dict[str, Dict]:
        return {rec['UUID']: rec for rec in library_df.to_dict(orient='records')}

    def _prepare_scrobbles(self, scrobble_df: pd.DataFrame) -> Tuple[List[Dict], List[str]]:
        recs = scrobble_df.to_dict(orient='records')
        texts = [f"{r['_n_artist']}{self._key_combo}{r['_n_album']}{self._key_combo}{r['_n_title']}" for r in recs]
        return recs, texts

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        cache = load_embed_cache()
        new_texts = [t for t in texts if t not in cache]
        if new_texts:
            embs = self._embedder.encode(
                new_texts,
                convert_to_numpy=True,
                batch_size=self._batch_size,
                show_progress_bar=False
            ).astype('float32')
            for t, emb in zip(new_texts, embs):
                cache[t] = emb
            save_embed_cache(cache)
        return np.vstack([cache[t] for t in texts])

    def _ann_candidates(self, vec: np.ndarray) -> List[str]:
        _, inds = self._index.search(vec.reshape(1, -1), self._ann_k)
        return [self._uuids[i] for i in inds[0] if i < len(self._uuids)]

    def _rerank(self, rec: Dict, candidate_ids: List[str], lib_lookup: Dict[str, Dict]) -> str:
        best_uuid, best_score = '<NO ASSOCIATED KEY>', 0.0
        for uid in candidate_ids:
            lib_rec = lib_lookup.get(uid)
            if lib_rec is None:
                continue
            score = self._scorer.score(rec, lib_rec, optimize=True)
            if score > best_score:
                best_score, best_uuid = score, uid
        return best_uuid if best_score >= self._threshold else '<NO ASSOCIATED KEY>'

    def link_scrobbles(self, library_df: pd.DataFrame, scrobble_df: pd.DataFrame) -> pd.DataFrame:
        lib_lookup = self._prepare_library(library_df)
        recs, texts = self._prepare_scrobbles(scrobble_df)
        embeddings = self._get_embeddings(texts)

        results: List[str] = []
        for rec, vec in zip(recs, embeddings):
            key = self._make_cache_key(rec)
            cached = self._cache.get(key)
            if cached:
                self._logger.trace(f"Cache hit {key}: {cached}", separator=self._SEPARATOR)
                results.append(cached)
                continue

            candidates = self._ann_candidates(vec)
            match = self._rerank(rec, candidates, lib_lookup)
            try:
                self._cache.set(key, match)
            except Exception as e:
                self._logger.error(f"Cache set failed for {key}: {e}", separator=self._SEPARATOR)
            results.append(match)

        scrobble_df['track_uuid'] = results
        self._cache.save()
        return scrobble_df
