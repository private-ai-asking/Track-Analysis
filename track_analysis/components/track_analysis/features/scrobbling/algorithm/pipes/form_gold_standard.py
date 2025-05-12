import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.scrobbling.algorithm.algorithm_context import AlgorithmContext


class FormGoldStandard(IPipe):
    """A pipe to form the gold standard based on x NN-Token accepts/rejects."""
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = "CacheBuilder.FormGoldStandard"

    def flow(self, ctx: AlgorithmContext) -> AlgorithmContext:
        auto_accepted: pd.DataFrame = ctx.auto_accepted_scrobbles
        auto_rejected: pd.DataFrame = ctx.auto_rejected_scrobbles
        confused_scrobbles: pd.DataFrame = ctx.confused_scrobbles

        self._logger.info(f"({len(auto_accepted)}a/{len(auto_rejected)}r/{len(confused_scrobbles)}u)", separator=self._separator)

        return ctx
