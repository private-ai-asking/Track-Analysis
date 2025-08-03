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

        main_df_write = pd.concat([context.generated_audio_info, context.loaded_audio_info_cache], ignore_index=True, sort=True)
        mfcc_df_write = pd.concat([context.generated_mfcc_audio_info, context.loaded_mfcc_info_cache], ignore_index=True, sort=True)
        key_progression_df_to_write = pd.concat([context.generated_key_progression_audio_info, context.loaded_key_progression_cache], ignore_index=True, sort=True)

        # Keep only columns defined in Header enum for main df
        allowed_columns = [h.value for h in Header]
        existing_columns = [col for col in allowed_columns if col in main_df_write.columns]
        main_df_write = main_df_write[existing_columns]

        total_main_df = main_df_write.shape[0]
        total_mfcc_df = mfcc_df_write.shape[0]
        total_key_progression_df = key_progression_df_to_write.shape[0]

        self._logger.info(f"Writing: {total_main_df} main rows.", separator=self._separator)
        main_df_write.to_csv(context.main_data_output_file_path, index=False)

        self._logger.info(f"Writing: {total_mfcc_df} mfcc rows.", separator=self._separator)
        mfcc_df_write.to_csv(context.mfcc_data_output_file_path, index=False)

        self._logger.info(f"Writing: {total_key_progression_df} key progression rows.", separator=self._separator)
        key_progression_df_to_write.to_csv(context.key_progression_output_file_path, index=False)

        self._logger.trace("Successfully written all data.", separator=self._separator)

        return context
