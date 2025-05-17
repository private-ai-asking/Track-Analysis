from pathlib import Path

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.scrobbling.algorithm.algorithm_context import AlgorithmContext


class FormGoldStandard(IPipe):
    """A pipe to form the gold standard based on x NN-Token accepts/rejects."""
    def __init__(self, logger: HoornLogger, max_gold_standard_entries: int, gold_standard_csv_path: Path):
        self._logger = logger
        self._separator = "CacheBuilder.FormGoldStandard"
        self._max_entries: int = max_gold_standard_entries

        self._csv_path: Path = gold_standard_csv_path

        # Define portion maxima
        self._accepted_max: int = int(max_gold_standard_entries * 0.25)
        self._rejected_max: int = int(max_gold_standard_entries * 0.25)
        self._uncertain_max: int = int(max_gold_standard_entries * 0.5)

    def flow(self, ctx: AlgorithmContext) -> AlgorithmContext:
        auto_accepted: pd.DataFrame = ctx.auto_accepted_scrobbles
        auto_rejected: pd.DataFrame = ctx.auto_rejected_scrobbles
        confused_scrobbles: pd.DataFrame = ctx.confused_scrobbles

        # Compute sample sizes capped by availability
        n_accept = min(len(auto_accepted), self._accepted_max)
        n_reject = min(len(auto_rejected), self._rejected_max)
        n_uncertain = min(len(confused_scrobbles), self._uncertain_max)

        # Sample up to that many rows
        accepted = auto_accepted.sample(n=n_accept, axis=0)
        rejected = auto_rejected.sample(n=n_reject, axis=0)
        uncertain = confused_scrobbles.sample(n=n_uncertain, axis=0)

        # Combine into one gold-standard set
        gold_standard_df: pd.DataFrame = pd.concat(
            [accepted, rejected, uncertain],
            ignore_index=True
        )

        # Restrict to only the desired columns and rename/add as requested
        keep_cols = ["_n_title", "_n_artist", "_n_album", "__key", "__predicted_uuid"]
        gold_standard_df = gold_standard_df.loc[:, keep_cols]
        gold_standard_df = gold_standard_df.rename(columns={"__predicted_uuid": "Predicted UUID"})
        gold_standard_df["Correct UUID"] = None

        # Log counts
        self._logger.info(
            f"accepted={len(accepted)}, "
            f"rejected={len(rejected)}, "
            f"uncertain={len(uncertain)}, "
            f"total={len(gold_standard_df)}",
            separator=self._separator
        )

        # Store back on context for downstream pipes
        ctx.gold_standard_df = gold_standard_df

        gold_standard_df.to_csv(self._csv_path)
        self._logger.info(f"Built gold standard at {self._csv_path}.", separator=self._separator)

        return ctx
