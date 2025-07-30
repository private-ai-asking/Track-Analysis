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

        all_paths = data.all_audio_file_paths
        total_disk = len(all_paths) or 1

        # If cache is empty, everything is new
        if data.loaded_audio_info_cache.shape[0] == 0:
            new_paths = all_paths
            cached_paths = []
        else:
            # Build a set of existing cache paths
            cache_paths = data.loaded_audio_info_cache[Header.Audio_Path.value].tolist()
            cache_set = set(cache_paths)

            # Partition into new vs. already-cached
            new_paths = [p for p in all_paths if str(p) not in cache_set]
            cached_paths = [p for p in all_paths if str(p) in cache_set]

        # Assign back into the pipeline context
        data.filtered_audio_file_paths = new_paths[:52]
        data.cached_audio_paths = cached_paths

        # Log progress
        self._logger.info(
            f"Audio files left to process: {len(data.filtered_audio_file_paths)}/{total_disk} "
            f"({round(len(data.filtered_audio_file_paths) / total_disk * 100, 2)}%)",
            separator=self._separator,
        )
        self._logger.debug(f"Left to process: {new_paths}", separator=self._separator)

        return data
