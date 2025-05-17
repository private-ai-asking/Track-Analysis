from pathlib import Path

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.scrobbling.algorithm.algorithm_context import AlgorithmContext


class ReportUncertainKeys(IPipe):
    """A pipe to report the status of the cache building."""
    def __init__(self, logger: HoornLogger, uncertain_keys_path: Path):
        self._logger = logger
        self._separator = "CacheBuilder.ReportUncertainKeys"
        self._uncertain_keys_path = uncertain_keys_path

    def flow(self, ctx: AlgorithmContext) -> AlgorithmContext:
        """Log any keys marked uncertain, in alphabetical order."""
        if not ctx.uncertain_keys:
            self._logger.info("No uncertain entries.", separator=self._separator)
        else:
            self._logger.info("Uncertain keys for review:", separator=self._separator)
            for key in sorted(ctx.uncertain_keys):
                self._logger.info(f"  {key}", separator=self._separator)

        ctx.confused_scrobbles.to_csv(self._uncertain_keys_path)

        return ctx
