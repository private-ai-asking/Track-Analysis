import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.scrobbling.algorithm.algorithm_context import AlgorithmContext
from track_analysis.components.track_analysis.features.scrobbling.scrobble_utility import ScrobbleUtility


class FilterExactMatches(IPipe):
    """Pipe to flatten the data."""
    def __init__(self, logger: HoornLogger, scrobble_utils: ScrobbleUtility):
        self._logger: HoornLogger = logger
        self._separator: str = "CacheBuilder.FilterExactMatches"
        self._scrobble_utils: ScrobbleUtility = scrobble_utils
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def flow(self, ctx: AlgorithmContext) -> AlgorithmContext:
        """Compute a unique key for each scrobble and drop duplicates."""
        """Map any scrobble-key exact matches to cache with full confidence."""
        self._logger.debug("Applying exact match.", separator=self._separator)

        scrobble_data_frame: pd.DataFrame = ctx.scrobble_data_frame.copy()
        scrobble_data_frame['uuid'] = scrobble_data_frame['__key'].map(ctx.library_lookup_key_to_uuid)

        for _, row in scrobble_data_frame[scrobble_data_frame['uuid'].notna()].iterrows():
            self._scrobble_utils.save_cache_item(
                key=row['__key'],
                uuid=row['uuid'],
                confidence_factor_percentage=100.0,
                library_data_frame=ctx.library_data_frame
            )

        ctx.scrobble_data_frame = scrobble_data_frame[scrobble_data_frame['uuid'].isna()].drop(columns=['uuid']).reset_index(drop=True)
        ctx.previous_pipe_description = "filtering exact matches"

        return ctx
