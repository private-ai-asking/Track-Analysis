import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.scrobble_linking.algorithm.algorithm_context import CacheBuildingAlgorithmContext
from track_analysis.components.track_analysis.features.scrobble_linking.utils.scrobble_utility import ScrobbleUtility


class ExtractUniqueEntries(IPipe):
    """Pipe to flatten the data."""
    def __init__(self, logger: HoornLogger, scrobble_utils: ScrobbleUtility):
        self._logger: HoornLogger = logger
        self._separator: str = "CacheBuilder.ExtractUniqueEntries"
        self._scrobble_utils: ScrobbleUtility = scrobble_utils
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def flow(self, ctx: CacheBuildingAlgorithmContext) -> CacheBuildingAlgorithmContext:
        """Compute a unique key for each scrobble and drop duplicates."""
        self._logger.debug("Extracting unique keys.", separator=self._separator)

        # Work directly on the original DF to avoid an unnecessary full copy
        df = ctx.scrobble_data_frame

        # 1) Early-drop raw duplicates on the columns that define the key
        df = df.drop_duplicates(
            subset=['_n_title', '_n_artist', '_n_album'],
            keep='first',
        )

        # 2) Compute the unique key via a list comprehension (faster than apply)
        keys = [
            self._scrobble_utils.compute_key(t, a, b)
            for t, a, b in zip(df['_n_title'], df['_n_artist'], df['_n_album'])
        ]
        df = df.assign(__key=keys)

        # 3) Final dedup on the computed key, reset the index in-place
        df.drop_duplicates(subset='__key', keep='first', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Write back into the context
        ctx.scrobble_data_frame = df
        ctx.previous_pipe_description = "extracting unique entries"
        return ctx
