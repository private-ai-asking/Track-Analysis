import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.scrobble_linking.algorithm.algorithm_context import CacheBuildingAlgorithmContext


class FilterExactMatches(IPipe):
    """Pipe to flatten the data by auto‐accepting exact key matches."""
    def __init__(self, logger: HoornLogger):
        self._logger: HoornLogger = logger
        self._separator: str = "CacheBuilder.FilterExactMatches"
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def flow(self, ctx: CacheBuildingAlgorithmContext) -> CacheBuildingAlgorithmContext:
        """Mark exact‐key matches as auto‐accepted (100% confidence), but don't persist yet."""
        self._logger.debug("Applying exact match.", separator=self._separator)

        # 1) copy and map in the library uuid
        df = ctx.scrobble_data_frame.copy()
        df['__predicted_uuid'] = df['__key'].map(ctx.library_lookup_key_to_uuid)

        # 2) split matched vs unmatched
        matched = df[df['__predicted_uuid'].notna()].copy()
        unmatched = df[df['__predicted_uuid'].isna()] \
            .drop(columns=['__predicted_uuid']) \
            .copy()

        # 3) annotate matched rows with 100% confidence
        if not matched.empty:
            matched['__confidence'] = 100.0

            # 4) append to the context bucket (or set it if empty)
            ctx.auto_accepted_scrobbles = (
                pd.concat([ctx.auto_accepted_scrobbles, matched], ignore_index=True)
                if ctx.auto_accepted_scrobbles is not None
                else matched.reset_index(drop=True)
            )

            self._logger.info(
                f"Queued {len(matched)} exact-match scrobbles for later caching.",
                separator=self._separator
            )

        # 5) pass the leftovers onward
        ctx.scrobble_data_frame = unmatched.reset_index(drop=True)
        ctx.previous_pipe_description = "filtering exact matches"
        return ctx
