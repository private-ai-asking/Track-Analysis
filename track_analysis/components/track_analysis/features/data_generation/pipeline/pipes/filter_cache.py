from typing import Dict, List, Tuple

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext


class FilterCache(IPipe):
    def __init__(self, logger: HoornLogger):
        self._separator = "BuildCSV.FilterCache"
        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        df: pd.DataFrame = data.loaded_audio_info_cache

        missing_data, missing_count = self._get_missing_headers(df, data.missing_headers_to_fill)

        self._logger.info(f"Found {missing_count} rows with missing data!", separator=self._separator)

        data.missing_headers = missing_data

        return data

    @staticmethod
    def _get_missing_headers(df: pd.DataFrame, headers_to_fill: List[Header]) -> Tuple[Dict[Header, List[str]], int]:
        missing_data = {}
        missing_count = 0
        headers_set = set(headers_to_fill)

        for col in df.columns:
            header = Header(col)
            if header in headers_set:
                null_mask = df[header.value].isna()
                if null_mask.any():
                    uuids = df.loc[null_mask, Header.UUID.value].tolist()
                    missing_data[header] = uuids
                    missing_count += len(uuids)

        return missing_data, missing_count
