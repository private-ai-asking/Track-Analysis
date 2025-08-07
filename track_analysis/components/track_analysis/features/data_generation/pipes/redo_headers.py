import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.builders.key_data_frames_builder import \
    KeyDataFramesBuilder
from track_analysis.components.track_analysis.features.data_generation.helpers.cache_updater import CacheUpdater
from track_analysis.components.track_analysis.features.data_generation.mappers.results_mapper import ResultsMapper
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    MFCC_FEATURES
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.feature_to_header_mapping import \
    HEADER_TO_FEATURE_MAPPING


class RedoHeaders(IPipe):
    """
    Orchestrator class that re-calculates specified features and merges them
    back into the cached data using a dedicated updater.
    """

    _SEPARATOR = "BuildCSV.RedoHeaders"

    def __init__(self, logger: HoornLogger, results_mapper: ResultsMapper, key_builder: KeyDataFramesBuilder, cache_updater: CacheUpdater):
        """
        Initializes the pipe.

        Args:
            logger (HoornLogger): The logger instance for logging messages.
            results_mapper (ResultsMapper): The mapper to structure raw feature data.
            key_builder (KeyDataFramesBuilder): The builder for key-related DataFrames.
            cache_updater (CacheUpdater): The centralized handler for updating cached DataFrames.
        """
        self._logger = logger
        self._results_mapper = results_mapper
        self._key_builder = key_builder
        self._cache_updater = cache_updater
        self._logger.trace("Successfully initialized pipe.", separator=self._SEPARATOR)

    def flow(self, context: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        if not context.headers_to_refill:
            self._logger.debug("No headers to refill.", separator=self._SEPARATOR)
            return context

        self._logger.info(f"Refilling {len(context.headers_to_refill)} headers.", separator=self._SEPARATOR)

        # === 1. Prepare for Processing ===
        audio_uuid_col = Header.UUID.value
        df_to_process = context.loaded_audio_info_cache
        key_prog_dfs = []

        features_to_request = [
            HEADER_TO_FEATURE_MAPPING[header]
            for header in context.headers_to_refill if header in HEADER_TO_FEATURE_MAPPING
        ]

        # Special handling for Key features, which affects other calculations
        key_related_headers = {Header.Key, Header.Start_Key, Header.End_Key}
        if key_related_headers.intersection(set(context.headers_to_refill)):
            self._logger.info("Recalculating Key-related features (Key, Start Key, End Key).", separator=self._SEPARATOR)
            raw_key_results = context.key_processor.extract_raw_keys(df_to_process)
            key_df, key_prog_dfs = self._key_builder.build(raw_key_results, df_to_process)

            df_to_process = df_to_process.merge(
                key_df, left_index=True, right_on='original_index', how='left'
            ).drop(columns=['original_index'])

        if Header.MFCC in context.headers_to_refill:
            features_to_request.extend(MFCC_FEATURES)

        # === 2. Process Features and Map Results ===
        calculated_features_df = context.main_processor.process_batch(df_to_process, features_to_request)

        # Create the "full" DataFrame that ResultsMapper expects
        full_results_df = pd.concat([
            df_to_process.reset_index(drop=True),
            calculated_features_df.drop(columns=[audio_uuid_col], errors='ignore').reset_index(drop=True)
        ], axis=1)

        # Directly call the build() method to get structured data
        updated_data = self._results_mapper.build(full_results_df, key_prog_dfs)

        # === 3. Merge Updates into Cached DataFrames using the dedicated updater ===
        self._cache_updater.update_main_info(context, updated_data.main_audio_info, audio_uuid_col)
        self._cache_updater.update_mfcc(context, updated_data.mfcc_audio_info, audio_uuid_col)
        self._cache_updater.update_key_progression(context, updated_data.key_progression_audio_info, audio_uuid_col)

        self._logger.info("Successfully refilled headers and updated the cache.", separator=self._SEPARATOR)
        return context
