from typing import List

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext


class RemoveInvalidCachedEntries(IPipe):
    def __init__(self, logger: HoornLogger):
        self._separator = "BuildCSV.RemoveInvalidCachedEntries"

        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        df: pd.DataFrame = data.loaded_audio_info_cache
        invalid_path_entries: List[str] = [str(p) for p in data.invalid_cached_paths]

        remove_df: pd.DataFrame = df.loc[
            df[Header.Audio_Path.value].isin(invalid_path_entries)
            ]

        data.loaded_audio_info_cache = data.loaded_audio_info_cache.drop(remove_df.index)

        self._logger.info(f"Removed {len(data.invalid_cached_paths)} invalid paths from the library!", separator=self._separator)
        return data
