import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext


class MakeCSV(IPipe):
    def __init__(self, logger: HoornLogger):
        self._separator = "BuildCSV.MakeCSVPipe"

        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, context: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        self._logger.trace("Writing data...", separator=self._separator)

        to_write = pd.concat(
            [context.generated_audio_info, context.loaded_audio_info_cache],
            ignore_index=True,
            sort=True
        )

        to_write.to_csv(context.main_data_output_file_path)

        self._logger.trace("Successfully written all data.", separator=self._separator)

        return context
