from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.scrobble_linking.algorithm.algorithm_context import \
    CacheBuildingAlgorithmContext
from track_analysis.components.track_analysis.features.scrobble_linking.embedding.embedding_searcher import EmbeddingSearcher
from track_analysis.components.track_analysis.features.scrobble_linking.embedding.evaluation.best_candidate_selector import \
    BestCandidateSelector
from track_analysis.components.track_analysis.features.scrobble_linking.embedding.evaluation.candidate_filter_interface import \
    CandidateEvaluatorInterface
from track_analysis.components.track_analysis.features.scrobble_linking.embedding.evaluation.confidence_assigner import \
    GaussianConfidenceAssigner
from track_analysis.components.track_analysis.features.scrobble_linking.embedding.evaluation.decision_evaluator import \
    DecisionEvaluator
from track_analysis.components.track_analysis.features.scrobble_linking.embedding.evaluation.field_threshold_evaluator import \
    FieldThresholdEvaluator
from track_analysis.components.track_analysis.features.scrobble_linking.model.candidate_model import CandidateModel, \
    DecisionBin
from track_analysis.components.track_analysis.features.scrobble_linking.model.scrabble_cache_algorithm_parameters import \
    ScrobbleCacheAlgorithmParameters
from track_analysis.components.track_analysis.library.configuration.model.configuration import \
    TrackAnalysisConfigurationModel


class FilterManualOverride(IPipe):
    """A pipe to filter out the tracks present in the manual override."""
    def __init__(self,
                 logger: HoornLogger,
                 embedding_searcher: EmbeddingSearcher, params: ScrobbleCacheAlgorithmParameters,
                 app_config: TrackAnalysisConfigurationModel):
        self._logger = logger
        self._separator = "CacheBuilder.FilterManualOverride"
        self._config = app_config
        self._override_path: Path = app_config.paths.manual_override
        self._embedding_searcher: EmbeddingSearcher = embedding_searcher

        self._token_accept_threshold = params.token_accept_threshold / 100.0
        self._confidence_accept_threshold = params.confidence_accept_threshold
        self._confidence_reject_threshold = params.confidence_reject_threshold

        self._candidate_evaluators: List[CandidateEvaluatorInterface] = [
            FieldThresholdEvaluator(logger, self._token_accept_threshold),
            BestCandidateSelector(logger),
            GaussianConfidenceAssigner(logger, params.gaussian_sigma),
            DecisionEvaluator(logger, self._confidence_accept_threshold, self._confidence_reject_threshold, self._token_accept_threshold)
        ]

    def flow(self, ctx: CacheBuildingAlgorithmContext) -> CacheBuildingAlgorithmContext:
        """Apply any manual overrides: accept, reject, or leave for later."""
        self._logger.debug("Applying manual override.", separator=self._separator)
        manual_override_lookup: Dict[str, str] = ctx.manual_override_lookup

        df = ctx.scrobble_data_frame.copy()
        df_override, df_remaining = self._split_overrides(df, manual_override_lookup)

        accepted, rejected = self._partition_by_override(df_override)

        if not accepted.empty:
            self._handle_accepted(accepted, ctx)
        if not rejected.empty:
            self._handle_rejected(rejected, ctx)

        return self._finalize_context(ctx, df_remaining)

    @staticmethod
    def _split_overrides(df: pd.DataFrame, lookup: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (overrides, remaining) based on keys in the lookup."""
        mask = df['__key'].isin(lookup)
        df_override  = df[mask].copy()
        df_override['override_uuid'] = df_override['__key'].map(lookup)
        df_remaining = df[~mask].copy()
        return df_override, df_remaining

    @staticmethod
    def _partition_by_override(df_override: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split override frame into accepted vs. rejected."""
        df_accepted = df_override[df_override['override_uuid'].notna()].copy()
        df_rejected = df_override[df_override['override_uuid'].isna()].copy()
        return df_accepted, df_rejected

    def _handle_accepted(self, df_accepted: pd.DataFrame, ctx: CacheBuildingAlgorithmContext) -> None:
        """Assign predictions for accepted scrobbles and update context."""
        df_processed = (
            df_accepted
            .assign(
                __predicted_uuid = df_accepted['override_uuid'],
                __confidence     = 100.0
            )
            .drop(columns=['override_uuid'])
        )

        # Merge back into context
        if ctx.auto_accepted_scrobbles is None:
            ctx.auto_accepted_scrobbles = df_processed.reset_index(drop=True)
        else:
            ctx.auto_accepted_scrobbles = pd.concat(
                [ctx.auto_accepted_scrobbles, df_processed],
                ignore_index=True
            )

        self._logger.info(
            f"Auto-accepted {len(df_processed)} scrobbles via manual override.",
            separator=self._separator
        )

    def _handle_rejected(self, df_rejected: pd.DataFrame, ctx: CacheBuildingAlgorithmContext) -> None:
        """Assign predictions for rejected scrobbles and update context."""
        df_processed = (
            df_rejected
            .assign(
                __predicted_uuid = None,
                __confidence     = 0.0
            )
            .drop(columns=['override_uuid'])
        )

        self._log_rejected_rows_that_are_predicted_to_be_accepted(df_processed)

        # Merge back into context
        if ctx.auto_rejected_scrobbles is None:
            ctx.auto_rejected_scrobbles = df_processed.reset_index(drop=True)
        else:
            ctx.auto_rejected_scrobbles = pd.concat(
                [ctx.auto_rejected_scrobbles, df_processed],
                ignore_index=True
            )

        self._logger.info(
            f"Auto-rejected {len(df_processed)} scrobbles via manual override.",
            separator=self._separator
        )

    def _log_rejected_rows_that_are_predicted_to_be_accepted(
            self,
            df: pd.DataFrame
    ) -> None:
        """Log each previously rejected row whose top FAISS candidate is now ACCEPT."""
        # 1) Build mask of rows to process
        mask = ~df['__key'].isin(self._config.additional_config.keys_to_be_ignored_in_cache_check)
        df_to_check = df.loc[mask]

        if df_to_check.empty:
            return

        # 2) Extract the columns for batch search
        titles = df_to_check['_n_title'].tolist()
        albums = df_to_check['_n_album'].tolist()
        artists = df_to_check['_n_artist'].tolist()

        # 3) Perform one big batch search
        all_candidates: List[List[CandidateModel]] = (
            self._embedding_searcher.search_batch(
                n_titles=titles,
                n_albums=albums,
                n_artists=artists,
                candidate_evaluators=self._candidate_evaluators
            )
        )

        # 4) Iterate only to log warnings (pure Python, no embedding calls)
        for key, candidates in zip(df_to_check['__key'], all_candidates):
            best_candidate = candidates[0]
            if best_candidate.decision_bin == DecisionBin.ACCEPT:
                self._logger.warning(
                    f"Row [{key}] should now be accepted according to the algorithm. "
                    "It will stay as is currently in the manual overwrite until you change it.\n"
                    f"Associated ID: [{best_candidate.uuid}]",
                    separator=self._separator
                )

    @staticmethod
    def _finalize_context(ctx: CacheBuildingAlgorithmContext, df_remaining: pd.DataFrame) -> CacheBuildingAlgorithmContext:
        """Reset the working frame and update the pipeline description."""
        ctx.scrobble_data_frame    = df_remaining.reset_index(drop=True)
        ctx.previous_pipe_description = "filtering manual override"
        return ctx
