from typing import Tuple, Dict, List, Optional

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.scrobbling.algorithm.algorithm_context import CacheBuildingAlgorithmContext
from track_analysis.components.track_analysis.features.scrobbling.embedding.embedding_searcher import EmbeddingSearcher
from track_analysis.components.track_analysis.features.scrobbling.embedding.evaluation.best_candidate_selector import \
    BestCandidateSelector
from track_analysis.components.track_analysis.features.scrobbling.embedding.evaluation.candidate_filter_interface import \
    CandidateEvaluatorInterface
from track_analysis.components.track_analysis.features.scrobbling.embedding.evaluation.confidence_assigner import \
    GaussianConfidenceAssigner
from track_analysis.components.track_analysis.features.scrobbling.embedding.evaluation.decision_evaluator import \
    DecisionEvaluator
from track_analysis.components.track_analysis.features.scrobbling.embedding.evaluation.field_threshold_evaluator import \
    FieldThresholdEvaluator
from track_analysis.components.track_analysis.features.scrobbling.model.candidate_model import CandidateModel, \
    DecisionBin


class NearestNeighborSearch(IPipe):
    """
    Performs nearest-neighbor search using combined embeddings,
    then enforces strict per-field token similarity before ranking.
    """
    def __init__(
            self,
            logger: HoornLogger,
            params,
            embedding_searcher: EmbeddingSearcher,
            test_mode: bool,
    ):
        self._logger = logger
        self._test_mode = test_mode
        self._searcher = embedding_searcher

        self._token_accept_threshold = params.token_accept_threshold / 100.0
        self._confidence_accept_threshold = params.confidence_accept_threshold
        self._confidence_reject_threshold = params.confidence_reject_threshold

        self._candidate_evaluators: List[CandidateEvaluatorInterface] = [
            FieldThresholdEvaluator(logger, self._token_accept_threshold),
            BestCandidateSelector(logger),
            GaussianConfidenceAssigner(logger, params.gaussian_sigma),
            DecisionEvaluator(logger, self._confidence_accept_threshold, self._confidence_reject_threshold, self._token_accept_threshold)
        ]

        # FAISS search params
        self._batch_size = params.batch_size
        self._top_k = params.top_k

        # Result buckets
        self._accept_keys: List[str] = []
        self._reject_keys: List[str] = []
        self._uncertain_keys: List[str] = []
        self._predictions: Dict[str, Optional[str]] = {}
        self._confidences: Dict[str, float] = {}

        self._separator: str = "CacheBuilder.NearestNeighbourSearch"

        self._logger.trace("Initialized NearestNeighborSearch.", separator=self._separator)

    def flow(self, ctx: CacheBuildingAlgorithmContext) -> CacheBuildingAlgorithmContext:
        scrobble_data = ctx.scrobble_data_frame
        if scrobble_data is None or scrobble_data.empty:
            self._logger.warning("No scrobbles to process.", separator=self._separator)
            return ctx

        n_titles, n_artists, n_albums = self._encode(scrobble_data)
        candidates: List[List[CandidateModel]] = self._searcher.search_batch(n_titles, n_albums, n_artists, candidate_evaluators=self._candidate_evaluators)

        self._handle_all(scrobble_data, candidates)
        self._finalize(ctx)
        return ctx

    @staticmethod
    def _encode(df) -> Tuple[List[str], List[str], List[str]]:
        return (
            df['_n_title'].tolist(),
            df['_n_artist'].tolist(),
            df['_n_album'].tolist(),
        )

    def _handle_all(self, scrobble_data: pd.DataFrame, all_candidates: List[List[CandidateModel]]) -> None:
        for idx, key in enumerate(scrobble_data['__key']):
            candidates: List[CandidateModel] = all_candidates[idx]

            if len(candidates) > 0:
                self._handle_candidate(key, candidates[0])
            else:
                self._handle_no_candidate(key)

    def _handle_no_candidate(self, key: str) -> None:
        # no candidates at all
        self._predictions[key] = None
        self._confidences[key] = 0.0

        if not self._test_mode:
            self._logger.debug(f"Auto-rejected {key} (confidence=0.00)")
        self._reject_keys.append(key)

    def _handle_candidate(self, key: str, candidate: CandidateModel) -> None:
        self._predictions[key] = candidate.uuid
        self._confidences[key] = candidate.associated_confidence

        if self._test_mode or candidate.decision_bin == DecisionBin.ACCEPT:
            self._accept_keys.append(key)
        if candidate.decision_bin == DecisionBin.REJECT:
            self._logger.debug(f"Rejected key \"{key}\"! (conf={candidate.associated_confidence},sim={candidate.combined_token_similarity})", separator=self._separator)
            self._reject_keys.append(key)
        if candidate.decision_bin == DecisionBin.UNCERTAIN:
            self._logger.debug(f"Marked key \"{key}\" as uncertain! (conf={candidate.associated_confidence},sim={candidate.combined_token_similarity})", separator=self._separator)
            self._uncertain_keys.append(key)

    def _finalize(self, ctx: CacheBuildingAlgorithmContext) -> None:
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
            f"Uncertain: {len(self._uncertain_keys)}",
            separator=self._separator
        )

