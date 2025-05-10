import pandas as pd
from typing import List, Optional, Dict
from rapidfuzz import fuzz

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import SimilarityScorer


class ScrobbleMatcher:
    """Used to match individual scrobbles to library data, with caching, optimized via SimilarityScorer."""

    def __init__(self,
                 logger: HoornLogger,
                 cache_builder: CacheBuilder,
                 key_combo: str = "||",
                 field_weights: Optional[Dict[str, float]] = None,
                 similarity_func = fuzz.token_sort_ratio,
                 threshold: float = 95.0
                 ):
        self._logger = logger
        self._cache_builder = cache_builder
        self._key_combo = key_combo
        # default weights
        weights = field_weights or {
            '_n_title': 0.5,
            '_n_artist': 0.3,
            '_n_album': 0.2
        }
        # initialize scorer
        self._scorer = SimilarityScorer(weights, logger, similarity_func, threshold)
        self._logger.trace("Successfully initialized.", separator="ScrobbleMatcher")

    def _make_cache_key(self, record: Dict) -> str:
        return f"{record['_n_artist']}{self._key_combo}{record['_n_album']}{self._key_combo}{record['_n_title']}"

    def link_scrobbles(self,
                       library_data: pd.DataFrame,
                       scrobble_data: pd.DataFrame
                       ) -> pd.DataFrame:
        # 1) materialize to pure-Python records
        lib_records = library_data.to_dict(orient='records')
        scr_records = scrobble_data.to_dict(orient='records')

        # 2) block library by artist
        blocks: Dict[str, List[Dict]] = {}
        for lib in lib_records:
            blocks.setdefault(lib['_n_artist'], []).append(lib)

        # 3) match each scrobble
        results: List[str] = []
        for scr in scr_records:
            key = self._make_cache_key(scr)
            cached = self._cache_builder.get(key)
            if cached is not None:
                self._logger.trace(f"Cache hit for key [{key}]: {cached}", separator="ScrobbleMatcher")
                results.append(cached)
                continue

            candidates = blocks.get(scr['_n_artist'], lib_records)
            best_uuid = None
            best_score = 0.0
            for lib in candidates:
                score = self._scorer.score(scr, lib, optimize=True)
                if score > best_score:
                    best_score = score
                    best_uuid = lib['UUID']

            uuid = best_uuid if best_score >= self._scorer.threshold else '<NO ASSOCIATED KEY>'
            try:
                self._cache_builder.set(key, uuid)
                self._logger.trace(f"Cache set for key [{key}]: {uuid}", separator="ScrobbleMatcher")
            except Exception as e:
                self._logger.error(f"Failed to set cache for key [{key}]: {e}", separator="ScrobbleMatcher")
            results.append(uuid)

        # attach back and save cache
        scrobble_data['track_uuid'] = results
        self._cache_builder.save()
        return scrobble_data
