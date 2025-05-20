from pathlib import Path
from typing import List

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import LibraryDataGenerationPipelineContext


class GetInvalidCache(IPipe):
    def __init__(self, logger: HoornLogger):
        self._separator = "BuildCSV.GetInvalidCachePipe"
        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        self._logger.trace("Validating cache entries...", separator=self._separator)

        if len(data.loaded_audio_info_cache) <= 0:
            data.invalid_cached_paths = []
            return data

        # Extract all cached paths as strings
        cache_paths_str = data.loaded_audio_info_cache[Header.Audio_Path.value].tolist()

        # 1) Detect duplicates in the cache
        path_series = pd.Series(cache_paths_str)
        dup_counts = path_series.value_counts()
        duplicates = dup_counts[dup_counts > 1].index.tolist()
        if duplicates:
            self._logger.warning(
                f"Duplicate cache entries detected for {len(duplicates)} paths: {duplicates[:10]}",
                separator=self._separator,
            )

        # 2) Detect invalid (non-existent or non-file) paths
        invalid_paths: List[Path] = []
        for path_str in set(cache_paths_str):
            path = Path(path_str)
            if not path.exists() or not path.is_file():
                self._logger.warning(f"Invalid path: {path}", separator=self._separator)
                invalid_paths.append(path)

        total_cached = len(cache_paths_str) or 1
        self._logger.info(
            f"Found number of invalid cached paths: {len(invalid_paths)}/{total_cached} "
            f"({round(len(invalid_paths)/total_cached*100, 2)}%)",
            separator=self._separator,
        )
        data.invalid_cached_paths = invalid_paths

        return data
