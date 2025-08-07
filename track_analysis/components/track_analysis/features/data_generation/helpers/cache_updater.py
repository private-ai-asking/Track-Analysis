import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext


class CacheUpdater:
    """
    A dedicated class for handling updates to the cached DataFrames within the pipeline context.
    This centralizes update logic to avoid duplication in pipeline steps.
    """
    _SEPARATOR = "CacheUpdater"

    def __init__(self, logger: HoornLogger):
        """
        Initializes the updater.

        Args:
            logger (HoornLogger): The logger instance for logging messages.
        """
        self._logger = logger

    def update_main_info(self, context: LibraryDataGenerationPipelineContext, updates_df: pd.DataFrame, index_col: str):
        """Updates the main audio info cache with new data."""
        if updates_df.empty:
            return

        self._logger.debug(f"Updating main audio info cache with {len(updates_df)} records.", separator=self._SEPARATOR)
        original_df_indexed = context.loaded_audio_info_cache.set_index(index_col)
        updates_df_indexed = updates_df.set_index(index_col)

        # Update only the columns that exist in the updates DataFrame
        cols_to_update = [col for col in updates_df_indexed.columns if col in original_df_indexed.columns]
        original_df_indexed.update(updates_df_indexed[cols_to_update])
        context.loaded_audio_info_cache = original_df_indexed.reset_index()
        self._logger.debug("Main audio info cache update complete.", separator=self._SEPARATOR)

    def update_mfcc(self, context: LibraryDataGenerationPipelineContext, updates_df: pd.DataFrame, index_col: str):
        """Updates the MFCC audio info cache with new data."""
        if updates_df.empty or context.loaded_mfcc_info_cache is None:
            return

        self._logger.debug(f"Updating MFCC cache with {len(updates_df)} records.", separator=self._SEPARATOR)
        original_df_indexed = context.loaded_mfcc_info_cache.set_index(index_col)
        updates_df_indexed = updates_df.set_index(index_col)

        # For MFCC, we can do a full update
        original_df_indexed.update(updates_df_indexed)
        context.loaded_mfcc_info_cache = original_df_indexed.reset_index()
        self._logger.debug("MFCC cache update complete.", separator=self._SEPARATOR)


    def update_key_progression(self, context: LibraryDataGenerationPipelineContext, updates_df: pd.DataFrame, index_col: str):
        """Updates the key progression cache by removing old data and appending new data."""
        if updates_df.empty or context.loaded_key_progression_cache is None:
            return

        self._logger.debug(f"Updating key progression cache with {len(updates_df)} records.", separator=self._SEPARATOR)

        # Strategy: Remove old data for the affected tracks and append the new data.
        uuids_to_update = updates_df[index_col].unique()
        original_df = context.loaded_key_progression_cache

        # Keep only the data for tracks that are NOT being updated
        filtered_original_df = original_df[~original_df[index_col].isin(uuids_to_update)]

        # Concatenate the old, unaffected data with the new data
        context.loaded_key_progression_cache = pd.concat([filtered_original_df, updates_df], ignore_index=True)
        self._logger.debug("Key progression cache update complete.", separator=self._SEPARATOR)
