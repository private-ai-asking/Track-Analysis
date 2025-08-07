from typing import List, Dict, Any

import pandas as pd

from track_analysis.components.md_common_python.py_common.algorithms.sequence.run_length_merger import StateRun
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature


class KeyProgressionDFBuilder:
    """Builds a tidy DataFrame from raw key progression results."""

    _KEY_PROGRESSION_COL = AudioDataFeature.KEY_PROGRESSION.name
    _UUID_COL = Header.UUID.value

    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = self.__class__.__name__

    def build(self, full_df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds the Key Progression DataFrame from the raw results.

        This method transforms the input DataFrame, where each row contains a list
        of key progression segments ('StateRun' objects), into a long-format DataFrame.
        Each segment from the original list becomes a separate row in the output,
        associated with its track UUID.

        Args:
            full_df: The input DataFrame containing track UUIDs and the
                     raw key progression data.

        Returns:
            A new DataFrame in a tidy format with columns for UUID, segment start,
            segment end, and the determined key for that segment. Returns an empty
            DataFrame if the input is empty or missing required columns.
        """
        if not self._is_input_valid(full_df):
            return pd.DataFrame()

        rows = self._generate_rows(full_df)

        return self._create_dataframe_from_rows(rows)

    def _is_input_valid(self, df: pd.DataFrame) -> bool:
        """Checks if the input DataFrame is valid for processing."""
        if df.empty:
            self._logger.warning(
                "Input DataFrame is empty. Skipping processing.",
                separator=self._separator
            )
            return False

        required_cols = {self._KEY_PROGRESSION_COL, self._UUID_COL}
        missing_cols = required_cols - set(df.columns)

        if missing_cols:
            self._logger.warning(
                f"Input DataFrame is missing required columns: {sorted(list(missing_cols))}. "
                "Skipping processing.",
                separator=self._separator
            )
            return False

        return True

    def _generate_rows(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Iterates through the DataFrame and transforms the data into a list of row dictionaries."""
        output_rows = []
        for row in df.itertuples(index=False):
            track_uuid = getattr(row, self._UUID_COL)
            progression_segments: List[StateRun] = getattr(row, self._KEY_PROGRESSION_COL)

            if not progression_segments:
                continue

            for segment in progression_segments:
                output_rows.append({
                    "Track UUID": track_uuid,
                    "Segment Start": segment.start_time,
                    "Segment End": segment.end_time,
                    "Segment Key": segment.state_label,
                })
        return output_rows

    @staticmethod
    def _create_dataframe_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """Creates and formats the final DataFrame from a list of rows."""
        if not rows:
            return pd.DataFrame()

        result_df = pd.DataFrame(rows)
        final_columns = ['Track UUID', 'Segment Start', 'Segment End', 'Segment Key']

        final_columns_exist = [col for col in final_columns if col in result_df.columns]

        return result_df[final_columns_exist]
