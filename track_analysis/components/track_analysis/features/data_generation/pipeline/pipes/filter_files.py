from pathlib import Path
from typing import List, Tuple

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import LibraryDataGenerationPipelineContext


class FilterFiles(IPipe):
    def __init__(self, logger: HoornLogger):
        self._separator = "BuildCSV.FilterFilesPipe"
        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        self._logger.trace("Filtering audio files...", separator=self._separator)

        # Extract all cached paths from the DataFrame
        cache_paths = data.loaded_audio_info_cache[Header.Audio_Path.value].tolist()
        cache_set = set(cache_paths)

        # New files: on disk but not in cache
        new_paths: List[Path] = [
            path for path in data.all_audio_file_paths
            if str(path) not in cache_set
        ]
        # Cached paths: on disk and in cache
        cached_paths: List[Path] = [
            path for path in data.all_audio_file_paths
            if str(path) in cache_set
        ]

        # Assign results to pipeline context
        data.filtered_audio_file_paths = new_paths
        data.cached_audio_paths = cached_paths

        total_disk = len(data.all_audio_file_paths) or 1
        self._logger.info(
            f"Audio files left to process: {len(new_paths)}/{total_disk} "
            f"({round(len(new_paths)/total_disk, 2)}%)",
            separator=self._separator,
        )
        self._logger.debug(f"Left to process: {new_paths}", separator=self._separator)

        return data
