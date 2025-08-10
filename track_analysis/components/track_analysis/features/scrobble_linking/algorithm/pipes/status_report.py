from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.scrobble_linking.algorithm.algorithm_context import CacheBuildingAlgorithmContext


class StatusReport(IPipe):
    """A pipe to report the status of the cache building."""
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = "CacheBuilder.StatusReport"

    def flow(self, ctx: CacheBuildingAlgorithmContext) -> CacheBuildingAlgorithmContext:
        scrobble_count: int = len(ctx.scrobble_data_frame)
        original: int = ctx.original_scrobble_count

        pct = (scrobble_count / original * 100) if original else 0
        self._logger.info(f"({scrobble_count}/{original}) [{pct:.2f}%] scrobbles after {ctx.previous_pipe_description}.", separator=self._separator)

        return ctx
