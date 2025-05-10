import numpy as np
import faiss
import pickle
from pathlib import Path
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import SimilarityScorer
from track_analysis.components.track_analysis.constants import DATA_DIRECTORY

# On-disk cache for scrobble embeddings
_EMBED_CACHE_PATH = Path(DATA_DIRECTORY) / "__internal__" / "scrobble_embed_cache.pkl"


def load_embed_cache() -> dict:
    try:
        with open(_EMBED_CACHE_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def save_embed_cache(cache: dict) -> None:
    _EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_EMBED_CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)


class ScrobbleMatcher:
    """Match scrobbles to library tracks using FAISS ANN + in-memory cached embeddings,
    exact-match shortcut, and batched search with serial fuzzy rerank."""

    def __init__(
            self,
            logger: HoornLogger,
            cache_builder: CacheBuilder,
            embedder: SentenceTransformer,
            faiss_index_path: Path,
            keys_path: Path,
            key_combo: str = "||",
            field_weights: dict = None,
            similarity_func = fuzz.token_sort_ratio,
            threshold: float = 95.0,
            ann_k: int = 5,
            batch_size: int = 64
    ):
        self._logger = logger
        self._cache = cache_builder
        self._embedder = embedder
        self._key_combo = key_combo
        self._threshold = threshold
        self._ann_k = ann_k
        self._batch_size = batch_size

        # load FAISS index and UUIDs
        self._index, self._uuids = self._load_faiss(faiss_index_path, keys_path)

        # init fuzzy scorer without chatty logging
        self._scorer = SimilarityScorer(field_weights or {}, self._logger, similarity_func, threshold)

        # load embed cache once
        self._embed_cache = load_embed_cache()

    def _load_faiss(self, index_path: Path, keys_path: Path):
        self._logger.info("Loading FAISS index and UUIDs...", separator="ScrobbleMatcher")
        index = faiss.read_index(str(index_path))
        with open(keys_path, 'rb') as f:
            uuids = pickle.load(f)
        assert index.ntotal == len(uuids), "Index size mismatch"
        return index, uuids

    def _build_exact_map(self, library_df):
        # Precompute exact lookup from combined string to UUID
        exact = {}
        combo = self._key_combo
        for rec in library_df.to_dict(orient='records'):
            key = f"{rec['_n_artist']}{combo}{rec['_n_album']}{combo}{rec['_n_title']}"
            exact[key] = rec['UUID']
        return exact

    def _get_embeddings(self, texts: list) -> np.ndarray:
        # Only compute new embeddings
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

    def _batch_ann(self, embeddings: np.ndarray) -> list:
        # Batch FAISS search
        _, indices = self._index.search(embeddings, self._ann_k)
        return [[self._uuids[i] for i in row if i < len(self._uuids)] for row in indices]

    def _rerank(self, rec: dict, candidates: list, lib_lookup: dict) -> str:
        best_uuid, best_score = '<NO ASSOCIATED KEY>', 0.0
        for uid in candidates:
            lib_rec = lib_lookup.get(uid)
            if not lib_rec:
                continue
            score = self._scorer.score(rec, lib_rec, optimize=True)
            if score > best_score:
                best_score, best_uuid = score, uid
        return best_uuid if best_score >= self._threshold else '<NO ASSOCIATED KEY>'

    def link_scrobbles(self, library_df, scrobble_df):
        # Prepare lookups
        lib_lookup = {r['UUID']: r for r in library_df.to_dict(orient='records')}
        exact_map = self._build_exact_map(library_df)
        recs = scrobble_df.to_dict(orient='records')
        texts = [f"{r['_n_artist']}{self._key_combo}{r['_n_album']}{self._key_combo}{r['_n_title']}" for r in recs]

        # Shortcut exact matches and cache hits
        results = []
        to_process = []
        for idx, (rec, txt) in enumerate(zip(recs, texts)):
            # 1) exact match
            if txt in exact_map:
                match = exact_map[txt]
                results.append(match)
                self._cache.set(txt, match)
            # 2) cache hit
            elif (cached := self._cache.get(txt)):
                results.append(cached)
            else:
                results.append(None)
                to_process.append(idx)

        if to_process:
            # batch embed and ann for those needing processing
            embeddings = self._get_embeddings(texts)
            candidates = self._batch_ann(embeddings)

            # serial rerank
            for idx in to_process:
                rec = recs[idx]
                cand = candidates[idx]
                match = self._rerank(rec, cand, lib_lookup)
                results[idx] = match
                self._cache.set(texts[idx], match)

        # attach results
        scrobble_df['track_uuid'] = results

        # persist caches once (embed + match)
        self._cache.save()
        save_embed_cache(self._embed_cache)

        return scrobble_df
