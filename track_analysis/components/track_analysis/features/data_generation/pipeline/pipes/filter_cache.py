from typing import Dict, List

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

        column_missing_dict: Dict[Header, List[str]] = {
            Header(col): df.loc[df[col].isna(), Header.UUID.value].tolist()
            for col in df.columns
            if Header(col) in data.missing_headers_to_fill and df[col].isna().any()
        }

        self._logger.info(f"Found rows with missing data!", separator=self._separator)

        data.missing_headers = column_missing_dict

        return data
