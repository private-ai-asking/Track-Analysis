import json

import pandas as pd
from pathlib import Path
from typing import Dict

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.scrobbling.algorithm.algorithm_context import AlgorithmContext


class FilterManualOverride(IPipe):
    """A pipe to filter out the tracks present in the manual override."""
    def __init__(self, logger: HoornLogger, manual_override_json_path: Path):
        self._logger = logger
        self._separator = "CacheBuilder.FilterManualOverride"
        self._override_path: Path = manual_override_json_path
        self._lookup = self._build_lookup()
        self._logger.trace(f"Loaded {len(self._lookup)} manual overrides.", separator=self._separator)

    def flow(self, ctx: AlgorithmContext) -> AlgorithmContext:
        """Apply any manual overrides: accept, reject, or leave for later."""
        self._logger.debug("Applying manual override.", separator=self._separator)

        df = ctx.scrobble_data_frame.copy()
        mask_override = df['__key'].isin(self._lookup)
        df_override  = df[mask_override].copy()
        df_remaining = df[~mask_override].copy()

        # map keys → override UUID (None means explicit reject)
        df_override['override_uuid'] = df_override['__key'].map(self._lookup)

        # split into accepted vs. rejected
        df_accepted = df_override[df_override['override_uuid'].notna()].copy()
        df_rejected = df_override[df_override['override_uuid'].isna()].copy()

        # --- Accepted: give them __predicted_uuid & __confidence, drop override_uuid ---
        if not df_accepted.empty:
            df_accepted = (
                df_accepted
                .assign(
                    __predicted_uuid = df_accepted['override_uuid'],
                    __confidence     = 100.0
                )
                .drop(columns=['override_uuid'])
            )
            ctx.auto_accepted_scrobbles = (
                pd.concat([ctx.auto_accepted_scrobbles, df_accepted], ignore_index=True)
                if ctx.auto_accepted_scrobbles is not None
                else df_accepted.reset_index(drop=True)
            )
            self._logger.info(
                f"Auto-accepted {len(df_accepted)} scrobbles via manual override.",
                separator=self._separator
            )

        # --- Rejected: still give them __predicted_uuid=None & __confidence=0, drop override_uuid ---
        if not df_rejected.empty:
            df_rejected = (
                df_rejected
                .assign(
                    __predicted_uuid = None,
                    __confidence     = 0.0
                )
                .drop(columns=['override_uuid'])
            )
            ctx.auto_rejected_scrobbles = (
                pd.concat([ctx.auto_rejected_scrobbles, df_rejected], ignore_index=True)
                if ctx.auto_rejected_scrobbles is not None
                else df_rejected.reset_index(drop=True)
            )
            self._logger.info(
                f"Auto-rejected {len(df_rejected)} scrobbles via manual override.",
                separator=self._separator
            )

        # leave the rest for downstream
        ctx.scrobble_data_frame = df_remaining.reset_index(drop=True)
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
