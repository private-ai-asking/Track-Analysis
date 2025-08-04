from typing import List

import pandas as pd


class KeyProgressionDFBuilder:
    def build(self, key_prog_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Builds the key progression DataFrame by concatenating partial results."""
        if not key_prog_dfs:
            return pd.DataFrame()

        return pd.concat(key_prog_dfs, ignore_index=True)
