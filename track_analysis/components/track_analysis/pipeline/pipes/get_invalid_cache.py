from pathlib import Path
from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.pipeline.pipeline_context import PipelineContextModel


class GetInvalidCache(IPipe):
    def __init__(self, logger: HoornLogger):
        self._separator = "GetInvalidCachePipe"

        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: PipelineContextModel) -> PipelineContextModel:
        self._logger.trace("Getting invalidated cache items...", separator=self._separator)

        cached_paths: List[Path] = [cached_audio_info.path for cached_audio_info in data.loaded_audio_info_cache]
        invalid_cached_paths: List[Path] = [cached_audio_info_path for cached_audio_info_path in cached_paths if not cached_audio_info_path.exists() or not cached_audio_info_path.is_file()]

        self._logger.info(f"Found number of invalid cached paths: {len(invalid_cached_paths)}/{len(cached_paths)}({round(len(invalid_cached_paths)/len(cached_paths)*100, 4)}%)", separator=self._separator)

        for path in invalid_cached_paths:
            self._logger.debug(f"Invalid path: {path}", separator=self._separator)

        data.invalid_cached_paths = invalid_cached_paths
        self._logger.trace(f"Successfully gotten all invalid cached items.", separator=self._separator)

        return data
