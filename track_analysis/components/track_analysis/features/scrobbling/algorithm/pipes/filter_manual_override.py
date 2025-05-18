import json

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.scrobbling.algorithm.algorithm_context import AlgorithmContext
from track_analysis.components.track_analysis.features.scrobbling.embedding.embedding_searcher import EmbeddingSearcher


class FilterManualOverride(IPipe):
    """A pipe to filter out the tracks present in the manual override."""
    def __init__(self, logger: HoornLogger, manual_override_json_path: Path, embedding_searcher: EmbeddingSearcher):
        self._logger = logger
        self._separator = "CacheBuilder.FilterManualOverride"
        self._override_path: Path = manual_override_json_path
        self._embedding_searcher: EmbeddingSearcher = embedding_searcher
        self._lookup = self._build_lookup()
        self._logger.trace(f"Loaded {len(self._lookup)} manual overrides.", separator=self._separator)

    def flow(self, ctx: AlgorithmContext) -> AlgorithmContext:
        """Apply any manual overrides: accept, reject, or leave for later."""
        self._logger.debug("Applying manual override.", separator=self._separator)

        df = ctx.scrobble_data_frame.copy()
        df_override, df_remaining = self._split_overrides(df)

        accepted, rejected = self._partition_by_override(df_override)

        if not accepted.empty:
            self._handle_accepted(accepted, ctx)
        if not rejected.empty:
            self._handle_rejected(rejected, ctx)

        return self._finalize_context(ctx, df_remaining)

    def _split_overrides(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (overrides, remaining) based on keys in the lookup."""
        mask = df['__key'].isin(self._lookup)
        df_override  = df[mask].copy()
        df_override['override_uuid'] = df_override['__key'].map(self._lookup)
        df_remaining = df[~mask].copy()
        return df_override, df_remaining

    @staticmethod
    def _partition_by_override(df_override: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split override frame into accepted vs. rejected."""
        df_accepted = df_override[df_override['override_uuid'].notna()].copy()
        df_rejected = df_override[df_override['override_uuid'].isna()].copy()
        return df_accepted, df_rejected

    def _handle_accepted(self, df_accepted: pd.DataFrame, ctx: AlgorithmContext) -> None:
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

    def _handle_rejected(self, df_rejected: pd.DataFrame, ctx: AlgorithmContext) -> None:
        """Assign predictions for rejected scrobbles and update context."""
        df_processed = (
            df_rejected
            .assign(
                __predicted_uuid = None,
                __confidence     = 0.0
            )
            .drop(columns=['override_uuid'])
        )

        # self._log_rejected_rows(df_processed)

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

    # def _log_rejected_rows(self, df: pd.DataFrame) -> None:
    #     """Temporary helper: print each rejected row for future processing."""
    #     for _, row in df.iterrows():
    #         self._embedding_searcher._search()

    @staticmethod
    def _finalize_context(ctx: AlgorithmContext, df_remaining: pd.DataFrame) -> AlgorithmContext:
        """Reset the working frame and update the pipeline description."""
        ctx.scrobble_data_frame    = df_remaining.reset_index(drop=True)
        ctx.previous_pipe_description = "filtering manual override"
        return ctx


    def _build_lookup(self) -> Dict[str, str]:
        if not self._override_path.is_file():
            self._logger.warning("Cannot load override cache, file is non-existent.", separator=self._separator)
            return {}
        if not self._override_path.name.endswith('.json'):
            self._logger.warning("Cannot load override cache, file is not json.", separator=self._separator)
            return {}

        with open(self._override_path, 'r') as f:
            data = json.load(f)
            return data
