from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.md_common_python.py_common.utils import (
    gaussian_exponential_kernel_confidence_percentage,
    SimilarityScorer,
    CollectionExtensions,
)
from track_analysis.components.track_analysis.features.scrobbling.algorithm.algorithm_context import AlgorithmContext


@dataclass
class _Candidate:
    lib_idx: int
    uuid: str
    distance: float
    combined_sim: float
    sim_title: float
    sim_artist: float
    sim_album: float


class NearestNeighborSearch(IPipe):
    """
    Performs nearest-neighbor search using combined embeddings,
    then enforces strict per-field token similarity before ranking.
    """
    def __init__(
            self,
            logger: HoornLogger,
            scrobble_util,            # ScrobbleUtility
            embedder: SentenceTransformer,
            params,
            test_mode: bool,
    ):
        self._logger = logger
        self._utils = scrobble_util
        self._embedder = embedder
        self._params = params
        self._test_mode = test_mode

        # Token-level scorer and thresholds
        self._scorer = SimilarityScorer(
            logger=logger,
            threshold=params.token_accept_threshold,
            field_weights=params.embed_weights,
        )
        self._token_thr = params.token_accept_threshold / 100.0
        self._accept_thr = params.confidence_accept_threshold
        self._reject_thr = params.confidence_reject_threshold
        self._sigma = params.gaussian_sigma

        # FAISS search params
        self._batch_size = params.batch_size
        self._top_k = params.top_k

        # Result buckets
        self._accept_keys: List[str] = []
        self._reject_keys: List[str] = []
        self._uncertain_keys: List[str] = []
        self._predictions: Dict[str, Optional[str]] = {}
        self._confidences: Dict[str, float] = {}

        # Pre-allocate FAISS output buffers
        n_queries = 10_000
        self._search_indices_buffer = np.empty((n_queries, self._top_k), dtype=np.int64)
        self._search_distances_buffer = np.empty((n_queries, self._top_k), dtype=np.float32)

        self._logger.trace("Initialized NearestNeighborSearch.")

    def flow(self, ctx: AlgorithmContext) -> AlgorithmContext:
        df = ctx.scrobble_data_frame
        if df is None or df.empty:
            self._logger.warning("No scrobbles to process.", separator="NearestNeighborSearch")
            return ctx

        embeddings = self._encode(df)
        indices, distances = self._search(embeddings, ctx)
        self._evaluate_all(df, distances, indices, ctx)
        self._finalize(ctx)
        return ctx

    def _encode(self, df) -> np.ndarray:
        return self._utils.build_combined_embeddings(
            df['_n_title'].tolist(),
            df['_n_artist'].tolist(),
            df['_n_album'].tolist(),
        )

    def _search(self, embeddings: np.ndarray, ctx: AlgorithmContext):
        if embeddings.dtype != np.float32 or not embeddings.flags["C_CONTIGUOUS"]:
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        nq = embeddings.shape[0]
        Dret, Iret = ctx.library_index.search(embeddings, self._top_k)
        self._search_distances_buffer[:nq] = Dret
        self._search_indices_buffer[:nq] = Iret
        return self._search_indices_buffer[:nq], self._search_distances_buffer[:nq]

    def _evaluate_all(self, df, distances: np.ndarray, indices: np.ndarray, ctx: AlgorithmContext) -> None:
        for idx, key in enumerate(df['__key']):
            rec = df.iloc[idx]
            candidates = self._gather_candidates(rec, indices[idx], distances[idx], ctx)
            best, max_token = self._find_best_candidate(candidates)
            if best is None:
                self._handle_no_candidate(key, candidates, max_token)
            else:
                self._handle_candidate(key, best)

    def _gather_candidates(self, rec, neighbors, dists, ctx) -> List[_Candidate]:
        rec_title, rec_artist, rec_album = rec["_n_title"], rec["_n_artist"], rec["_n_album"]
        score_fn = self._scorer.score
        lib_keys = ctx.library_keys
        n_lib = len(lib_keys)

        candidates: List[_Candidate] = []
        for lib_idx, distance in zip(neighbors, dists):
            if not (0 <= lib_idx < n_lib):
                continue
            uuid = lib_keys[lib_idx]
            row = ctx.library_row_lookup.get(uuid)
            if not row:
                continue

            combined_sim = score_fn(
                {"title": rec_title, "artist": rec_artist, "album": rec_album},
                {"title": row["_n_title"], "artist": row["_n_artist"], "album": row["_n_album"]},
                optimize=True
            ) / 100.0

            sim_title  = fuzz.ratio(rec_title,  row["_n_title"])  / 100.0
            sim_artist = fuzz.ratio(rec_artist, row["_n_artist"]) / 100.0
            sim_album  = fuzz.ratio(rec_album,  row["_n_album"])  / 100.0

            candidates.append(_Candidate(
                lib_idx      = lib_idx,
                uuid         = uuid,
                distance     = float(distance),
                combined_sim = combined_sim,
                sim_title    = sim_title,
                sim_artist   = sim_artist,
                sim_album    = sim_album,
            ))
        return candidates

    def _find_best_candidate(
            self, candidates: List[_Candidate]
    ) -> Tuple[Optional[_Candidate], float]:
        best, max_token = None, 0.0
        for cand in candidates:
            max_token = max(max_token, cand.combined_sim)
            if self._passes_field_threshold(cand) and (
                    best is None or cand.combined_sim > best.combined_sim
            ):
                best = cand
        return best, max_token

    def _passes_field_threshold(self, cand: _Candidate) -> bool:
        return (
                cand.sim_title  >= self._token_thr and
                cand.sim_artist >= self._token_thr and
                cand.sim_album  >= self._token_thr
        )

    def _handle_no_candidate(
            self,
            key: str,
            candidates: List[_Candidate],
            max_token: float
    ) -> None:
        # 1) Fallback prediction: choose the highest-scoring candidate if any
        if candidates:
            fallback = max(candidates, key=lambda c: c.combined_sim)
            self._predictions[key] = fallback.uuid
        else:
            # no candidates at all
            self._predictions[key] = None

        # 2) Confidence is always zero here
        self._confidences[key] = 0.0

        # 3) Decide bucket
        if max_token >= self._token_thr:
            self._uncertain_keys.append(key)
            self._logger.debug(f"Marked uncertain: {key}")
        else:
            if not self._test_mode:
                self._logger.debug(f"Auto-rejected {key} (confidence=0.00)")
            self._reject_keys.append(key)

    def _handle_candidate(self, key: str, cand: _Candidate) -> None:
        confidence = gaussian_exponential_kernel_confidence_percentage(
            cand.distance, sigma=self._sigma
        )
        self._predictions[key] = cand.uuid
        self._confidences[key] = confidence

        accept, reject, uncertain = self._decision(confidence, cand.combined_sim)
        if self._test_mode or accept:
            self._accept_keys.append(key)
        if reject:
            self._reject_keys.append(key)
        if uncertain:
            self._uncertain_keys.append(key)

    def _decision(self, confidence: float, token_sim: float) -> Tuple[bool, bool, bool]:
        accept = (confidence >= self._accept_thr) and (token_sim >= self._token_thr)
        reject = ((confidence >= self._accept_thr) and (token_sim < self._token_thr)) \
                 or (confidence <= self._reject_thr)
        uncertain = not accept and not reject

        if CollectionExtensions.more_than(lambda x: x, [accept, reject, uncertain], 1):
            self._logger.warning(
                f"Multiple decision states: accept={accept}, reject={reject}, uncertain={uncertain}"
            )
        return accept, reject, uncertain

    def _finalize(self, ctx: AlgorithmContext) -> None:
        df = ctx.scrobble_data_frame.copy()
        df['__confidence']     = df['__key'].map(self._confidences).fillna(0.0)
        df['__predicted_uuid'] = df['__key'].map(self._predictions)

        # slice out new frames
        new_accepted = df[df['__key'].isin(self._accept_keys)].reset_index(drop=True)
        new_rejected = df[df['__key'].isin(self._reject_keys)].reset_index(drop=True)
        new_confused = df[df['__key'].isin(self._uncertain_keys)].reset_index(drop=True)

        # append or set
        ctx.auto_accepted_scrobbles = (
            pd.concat([ctx.auto_accepted_scrobbles, new_accepted], ignore_index=True)
            if ctx.auto_accepted_scrobbles is not None
            else new_accepted
        )
        ctx.auto_rejected_scrobbles = (
            pd.concat([ctx.auto_rejected_scrobbles, new_rejected], ignore_index=True)
            if ctx.auto_rejected_scrobbles is not None
            else new_rejected
        )
        ctx.confused_scrobbles = (
            pd.concat([ctx.confused_scrobbles, new_confused], ignore_index=True)
            if ctx.confused_scrobbles is not None
            else new_confused
        )

        # leave only the “uncertain” for downstream
        ctx.scrobble_data_frame = new_confused.copy()
        ctx.uncertain_keys = self._uncertain_keys
        ctx.previous_pipe_description = 'nearest neighbour search'

        self._logger.info(
            f"NN complete. Accepted: {len(self._accept_keys)}, "
            f"Rejected: {len(self._reject_keys)}, "
            f"Uncertain: {len(self._uncertain_keys)}"
        )

