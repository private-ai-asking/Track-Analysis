import pandas as pd
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
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

        # Keep only columns defined in Header enum
        allowed_columns = [h.value for h in Header]
        existing_columns = [col for col in allowed_columns if col in to_write.columns]
        to_write = to_write[existing_columns]

        to_write.to_csv(context.main_data_output_file_path, index=False)

        self._logger.trace("Successfully written all data.", separator=self._separator)

        return context
