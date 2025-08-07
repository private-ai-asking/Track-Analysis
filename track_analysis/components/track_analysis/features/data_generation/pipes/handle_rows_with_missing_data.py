from pathlib import Path

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.helpers.cache_updater import CacheUpdater
from track_analysis.components.track_analysis.features.data_generation.mappers.results_mapper import ResultsMapper
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.feature_to_header_mapping import \
    HEADER_TO_FEATURE_MAPPING
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext


class FillMissingHeadersPipe(IPipe):
    """
    Fills rows with missing data for specified headers using a delegated processing strategy.
    This pipe identifies headers with missing values, recalculates them, and merges the
    updated data back into the main cache.
    """

    _SEPARATOR = "BuildCSV.FillMissingHeaders"

    def __init__(self, logger: HoornLogger, results_mapper: ResultsMapper, cache_updater: CacheUpdater):
        """
        Initializes the pipe.

        Args:
            logger (HoornLogger): The logger instance for logging messages.
            results_mapper (ResultsMapper): The mapper to structure raw feature data.
            cache_updater (CacheUpdater): The centralized handler for updating cached DataFrames.
        """
        self._logger = logger
        self._results_mapper = results_mapper
        self._cache_updater = cache_updater
        self._logger.trace("Successfully initialized pipe.", separator=self._SEPARATOR)

    def flow(self, context: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        """
        Executes the logic to fill missing header data.

        Args:
            context (LibraryDataGenerationPipelineContext): The pipeline context.

        Returns:
            LibraryDataGenerationPipelineContext: The updated pipeline context.
        """
        if not context.missing_headers:
            self._logger.debug("No headers with missing data to fill.", separator=self._SEPARATOR)
            return context

        headers_to_fill = set(context.missing_headers.keys())
        self._logger.info(f"Found {len(headers_to_fill)} headers with missing data.", separator=self._SEPARATOR)

        # --- 1. Handle Special Cases First ---
        key_headers = {Header.Start_Key, Header.End_Key, Header.Key}
        if key_headers.intersection(headers_to_fill):
            self._handle_segment_keys_fill(context)
            headers_to_fill.difference_update(key_headers)

        # --- 2. Handle Standard Audio Features via Main Processor ---
        if not headers_to_fill:
            self._logger.info("All missing headers were handled by specialized processors.", separator=self._SEPARATOR)
            return context

        self._logger.info(f"Refilling {len(headers_to_fill)} standard headers via main processor.", separator=self._SEPARATOR)

        uuids_to_process = {uuid for header in headers_to_fill for uuid in context.missing_headers.get(header, [])}
        if not uuids_to_process:
            self._logger.warning("Standard headers need filling, but no associated UUIDs found.", separator=self._SEPARATOR)
            return context

        uuid_col = Header.UUID.value
        original_df = context.loaded_audio_info_cache
        df_to_process = original_df[original_df[uuid_col].isin(list(uuids_to_process))]

        if df_to_process.empty:
            self._logger.info("No matching rows found in the cache for the given UUIDs.", separator=self._SEPARATOR)
            return context

        features_to_request = [HEADER_TO_FEATURE_MAPPING[header] for header in headers_to_fill if header in HEADER_TO_FEATURE_MAPPING]

        # Process features and map results using the ResultsMapper
        refilled_features_df = context.main_processor.process_batch(df_to_process, features_to_request)

        full_results_df = pd.concat([
            df_to_process.reset_index(drop=True),
            refilled_features_df.drop(columns=[uuid_col], errors='ignore').reset_index(drop=True)
        ], axis=1)

        # Directly call the build() method to get structured data
        updated_data = self._results_mapper.build(full_results_df, key_prog_dfs=[])

        # --- 3. Merge Updates into Cached DataFrame using the dedicated updater ---
        self._cache_updater.update_main_info(context, updated_data.main_audio_info, uuid_col)

        self._logger.info("Successfully filled standard headers and updated the cache.", separator=self._SEPARATOR)
        return context

    def _load_key_segments_df(self, key_csv_path: Path, uuids: set) -> pd.DataFrame | None:
        """Loads and filters the key progression CSV, returning None if not found or empty."""
        if not key_csv_path.exists():
            self._logger.warning(f"Key progression CSV not found at {key_csv_path}; cannot fill keys.",
                                 separator=self._SEPARATOR)
            return None

        segments_df = pd.read_csv(key_csv_path)
        filtered_segments = segments_df[segments_df["Track UUID"].isin(uuids)]

        if filtered_segments.empty:
            self._logger.info("No key segments found for the given UUIDs.", separator=self._SEPARATOR)
            return None

        return filtered_segments

    def _handle_segment_keys_fill(self, context: LibraryDataGenerationPipelineContext):
        """Fills missing Start Key and End Key from the key progression CSV using a vectorized approach."""
        start_key_uuids = set(context.missing_headers.get(Header.Start_Key, []))
        end_key_uuids = set(context.missing_headers.get(Header.End_Key, []))
        all_uuids = start_key_uuids | end_key_uuids

        if not all_uuids:
            return

        self._logger.info("Filling missing Start/End Keys...", separator=self._SEPARATOR)
        segments_df = self._load_key_segments_df(context.key_progression_output_file_path, all_uuids)
        if segments_df is None:
            return

        # Find the first and last segments for each track efficiently
        first_segments = segments_df.loc[segments_df.groupby("Track UUID")["Segment Start"].idxmin()]
        last_segments = segments_df.loc[segments_df.groupby("Track UUID")["Segment End"].idxmax()]

        # Create a mapping from UUID to the corresponding key
        start_key_map = first_segments.set_index("Track UUID")["Segment Key"]
        end_key_map = last_segments.set_index("Track UUID")["Segment Key"]

        # Apply the updates to the main DataFrame
        df = context.loaded_audio_info_cache
        uuid_col = Header.UUID.value

        # Create boolean masks for rows that need updating
        start_mask = df[uuid_col].isin(start_key_uuids)
        end_mask = df[uuid_col].isin(end_key_uuids)

        # Use the map to apply the new keys based on UUID
        df.loc[start_mask, Header.Start_Key.value] = df.loc[start_mask, uuid_col].map(start_key_map)
        df.loc[end_mask, Header.End_Key.value] = df.loc[end_mask, uuid_col].map(end_key_map)

        self._logger.info(f"Finished filling start/end keys for {len(segments_df['Track UUID'].unique())} tracks.",
                          separator=self._SEPARATOR)
