from typing import List

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.mappers.results_mapper import ResultsMapper, \
    MappedAudioData
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    KEY_PROGRESSION_UNIQUE_FILE_FEATURES, MFCC_UNIQUE_FILE_FEATURES, AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.feature_to_header_mapping import \
    HEADER_TO_FEATURE_MAPPING


class CacheUpdater:
    """
    A dedicated class for handling updates to the cached DataFrames within the pipeline context.
    This centralizes update logic to avoid duplication in pipeline steps.
    """
    _SEPARATOR = "CacheUpdater"

    def __init__(self, logger: HoornLogger, results_mapper: ResultsMapper):
        """
        Initializes the updater.

        Args:
            logger (HoornLogger): The logger instance for logging messages.
            results_mapper (ResultsMapper): The results mapper instance.
        """
        self._logger = logger
        self._results_mapper = results_mapper
        self._uuid_col = Header.UUID.value

    def update_cache(self, context: LibraryDataGenerationPipelineContext, uuids_to_process: List[str] | None, headers_to_update: List[Header]) -> LibraryDataGenerationPipelineContext:
        """
        Updates the cache's headers you provide with retrieved audio features.

        Arguments:
            context: The current pipeline context.
            uuids_to_process: The list of UUIDs to process. Set to none to process everything.
            headers_to_update: The list of headers to update.
        """

        original_df = context.loaded_audio_info_cache

        if uuids_to_process is not None:
            df_to_process = original_df[original_df[self._uuid_col].isin(list(uuids_to_process))]
        else:
            df_to_process = original_df

        if df_to_process.empty:
            self._logger.info("No matching rows found in the cache for the given UUIDs.", separator=self._SEPARATOR)
            return context

        features_to_request = self._get_features_to_request(headers_to_update)
        updated_features_df = context.main_processor.process_batch(df_to_process, features_to_request)

        full_results_df = self._get_full_results_df(original_df, updated_features_df)
        updated_data = self._results_mapper.build(full_results_df)
        self._update_data(context, updated_data)

        return context

    def _get_full_results_df(self, original_df: pd.DataFrame, updated_features_df: pd.DataFrame) -> pd.DataFrame:
        full_results_df = pd.concat([
            original_df.reset_index(drop=True),
            updated_features_df.drop(columns=[self._uuid_col], errors='ignore').reset_index(drop=True)
        ], axis=1)

        return full_results_df

    def _update_data(self, context: LibraryDataGenerationPipelineContext, updated_data: MappedAudioData) -> None:
        self._update_main_info(context, updated_data.main_audio_info)
        self._update_key_progression(context, updated_data.key_progression_audio_info)
        self._update_mfcc(context, updated_data.mfcc_audio_info)

    @staticmethod
    def _get_features_to_request(headers_to_update: List[Header]) -> List[AudioDataFeature]:
        features_to_request = [HEADER_TO_FEATURE_MAPPING[header] for header in headers_to_update if header in HEADER_TO_FEATURE_MAPPING]
        if Header.Key_Progression in headers_to_update:
            features_to_request.extend(KEY_PROGRESSION_UNIQUE_FILE_FEATURES)

        if Header.MFCC in headers_to_update:
            features_to_request.extend(MFCC_UNIQUE_FILE_FEATURES)

        return features_to_request

    def _update_main_info(self, context: LibraryDataGenerationPipelineContext, updates_df: pd.DataFrame):
        """Updates the main audio info cache with new data."""
        if updates_df.empty:
            return

        self._logger.debug(f"Updating main audio info cache with {len(updates_df)} records.", separator=self._SEPARATOR)
        original_df_indexed = context.loaded_audio_info_cache.set_index(self._uuid_col)
        updates_df_indexed = updates_df.set_index(self._uuid_col)

        # Update only the columns that exist in the updates DataFrame
        cols_to_update = [col for col in updates_df_indexed.columns if col in original_df_indexed.columns]
        original_df_indexed.update(updates_df_indexed[cols_to_update])
        context.loaded_audio_info_cache = original_df_indexed.reset_index()
        self._logger.debug("Main audio info cache update complete.", separator=self._SEPARATOR)

    def _update_mfcc(self, context: LibraryDataGenerationPipelineContext, updates_df: pd.DataFrame):
        """Updates the MFCC audio info cache with new data."""
        if updates_df.empty or context.loaded_mfcc_info_cache is None:
            return

        self._logger.debug(f"Updating MFCC cache with {len(updates_df)} records.", separator=self._SEPARATOR)
        original_df_indexed = context.loaded_mfcc_info_cache.set_index(self._uuid_col)
        updates_df_indexed = updates_df.set_index(self._uuid_col)

        # For MFCC, we can do a full update
        original_df_indexed.update(updates_df_indexed)
        context.loaded_mfcc_info_cache = original_df_indexed.reset_index()
        self._logger.debug("MFCC cache update complete.", separator=self._SEPARATOR)


    def _update_key_progression(self, context: LibraryDataGenerationPipelineContext, updates_df: pd.DataFrame):
        """Updates the key progression cache by removing old data and appending new data."""
        if updates_df.empty or context.loaded_key_progression_cache is None:
            return

        self._logger.debug(f"Updating key progression cache with {len(updates_df)} records.", separator=self._SEPARATOR)

        # Strategy: Remove old data for the affected tracks and append the new data.
        uuids_to_update = updates_df[self._uuid_col].unique()
        original_df = context.loaded_key_progression_cache

        # Keep only the data for tracks that are NOT being updated
        filtered_original_df = original_df[~original_df[self._uuid_col].isin(uuids_to_update)]

        # Concatenate the old, unaffected data with the new data
        context.loaded_key_progression_cache = pd.concat([filtered_original_df, updates_df], ignore_index=True)
        self._logger.debug("Key progression cache update complete.", separator=self._SEPARATOR)
