from pathlib import Path
from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.pipeline.pipeline_context import PipelineContextModel


class FilterFiles(IPipe):
    def __init__(self, logger: HoornLogger):
        self._separator = "FilterFilesPipe"

        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def _gather_matches(self, paths1: List[Path], paths2: List[Path]) -> List[Path]:
        matches = []

        for path1 in paths1:
            for path2 in paths2:
                if path1 == path2:
                    matches.append(path1)
                    break

        return matches

    def flow(self, data: PipelineContextModel) -> PipelineContextModel:
        self._logger.trace("Filtering audio files...", separator=self._separator)

        to_exclude: List[Path] = self._gather_matches([cached_audio_info.path for cached_audio_info in data.loaded_audio_info_cache], data.all_audio_file_paths)
        filtered: List[Path] = [audio_file_path for audio_file_path in data.all_audio_file_paths if audio_file_path not in to_exclude]

        data.filtered_audio_file_paths = filtered

        self._logger.info(
            f"Audio files left to process: {len(data.filtered_audio_file_paths)}/{len(data.all_audio_file_paths)}({round(len(data.filtered_audio_file_paths)/len(data.all_audio_file_paths), 2)}%)",
            separator=self._separator,
        )
        self._logger.debug(f"Left to process: {filtered}", separator=self._separator)

        return data
