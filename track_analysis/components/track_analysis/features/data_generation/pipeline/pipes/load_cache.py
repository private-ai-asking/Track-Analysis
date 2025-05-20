import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext


class LoadCache(IPipe):
    def __init__(self, logger: HoornLogger):
        self._separator = "BuildCSV.LoadCachePipe"

        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        self._logger.trace("Loading cache if existing...", separator=self._separator)

        if not data.main_data_output_file_path.exists() or not data.main_data_output_file_path.is_file():
            return data

        self._logger.trace("Cache exists, proceeding to load...", separator=self._separator)

        cached_data: pd.DataFrame = pd.read_csv(data.main_data_output_file_path, header=0)

        num_lines = cached_data.shape[0]

        self._logger.info(f"Cache loaded. Number of records: {num_lines}", separator=self._separator)

        data.loaded_audio_info_cache = cached_data

        return data
