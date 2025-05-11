import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.scrobbling.algorithm.algorithm_context import AlgorithmContext
from track_analysis.components.track_analysis.features.scrobbling.scrobble_utility import ScrobbleUtility


class ExtractUniqueEntries(IPipe):
    """Pipe to flatten the data."""
    def __init__(self, logger: HoornLogger, scrobble_utils: ScrobbleUtility):
        self._logger: HoornLogger = logger
        self._separator: str = "CacheBuilder.ExtractUniqueEntries"
        self._scrobble_utils: ScrobbleUtility = scrobble_utils
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def flow(self, ctx: AlgorithmContext) -> AlgorithmContext:
        """Compute a unique key for each scrobble and drop duplicates."""
        self._logger.debug("Extracting unique keys.", separator=self._separator)
        scrobble_data_frame: pd.DataFrame = ctx.scrobble_data_frame.copy()
        scrobble_data_frame['__key'] = scrobble_data_frame.apply(
            lambda r: self._scrobble_utils.compute_key(r['_n_title'], r['_n_artist'], r['_n_album']),
            axis=1
        )

        ctx.scrobble_data_frame = scrobble_data_frame.drop_duplicates('__key').reset_index(drop=True)
        ctx.previous_pipe_description = "extracting unique entries"

        return ctx
