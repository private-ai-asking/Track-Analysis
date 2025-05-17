from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.scrobbling.algorithm.algorithm_context import AlgorithmContext
from track_analysis.components.track_analysis.features.scrobbling.cache_helper import ScrobbleCacheHelper


class StoreInCache(IPipe):
    """Persist any auto‐buckets into the scrobble cache."""
    def __init__(self, logger: HoornLogger, cache_helper: ScrobbleCacheHelper, test_mode: bool = False):
        self._logger = logger
        self._separator = "CacheBuilder.StoreInCache"
        self._cache = cache_helper
        self._test = test_mode
        self._logger.trace(
            f"Initialized (test_mode={self._test}).",
            separator=self._separator
        )

    def flow(self, ctx: AlgorithmContext) -> AlgorithmContext:
        self._logger.debug("Storing auto‐accepted scrobbles to cache…", separator=self._separator)

        def _persist(df, label: str):
            """Helper to write each row in df via the cache helper."""
            for _, row in df.iterrows():
                self._cache.save_cache_item(
                    key=row["__key"],
                    uuid=row.get("__predicted_uuid"),  # may be None for rejects
                    confidence_factor_percentage=row["__confidence"]
                )
            self._logger.info(
                f"Persisted {len(df)} {label}.",
                separator=self._separator
            )

        # --- always store auto‐accepted scrobbles ---
        if ctx.auto_accepted_scrobbles is not None and not ctx.auto_accepted_scrobbles.empty:
            _persist(ctx.auto_accepted_scrobbles, "auto‐accepted scrobbles")

        # --- in test mode, also store rejects & confused ---
        if self._test:
            if ctx.auto_rejected_scrobbles is not None and not ctx.auto_rejected_scrobbles.empty:
                self._logger.debug("Storing auto‐rejected scrobbles to cache (test mode)…", separator=self._separator)
                _persist(ctx.auto_rejected_scrobbles, "auto‐rejected scrobbles")

            if ctx.confused_scrobbles is not None and not ctx.confused_scrobbles.empty:
                self._logger.debug("Storing confused scrobbles to cache (test mode)…", separator=self._separator)
                _persist(ctx.confused_scrobbles, "confused scrobbles")

        # no changes to ctx.scrobble_data_frame; everything persisted is already
        # removed by upstream pipes. We just record that we’ve run this step.
        ctx.previous_pipe_description = "storing in cache"
        return ctx
