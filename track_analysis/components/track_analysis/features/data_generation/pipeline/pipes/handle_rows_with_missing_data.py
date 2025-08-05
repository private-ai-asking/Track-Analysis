from pathlib import Path

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.audio_calculation.feature_to_header_mapping import \
    HEADER_TO_FEATURE_MAPPING, FEATURE_TO_HEADER_MAPPING
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext


class FillMissingHeadersPipe(IPipe):
    """
    Fills rows with missing data for specified headers using a delegated processing strategy.

    This pipe identifies headers with missing values and routes them to the appropriate
    handler. Standard audio features are processed in a batch by a main processor,
    while special cases (e.g., Bit Depth, Segment Keys) are handled by dedicated
    methods. The main DataFrame is then updated with the new data.
    """

    _SEPARATOR = "BuildCSV.FillMissingHeaders"

    def __init__(self, logger: HoornLogger):
        """
        Initializes the pipe.

        Args:
            logger (HoornLogger): The logger instance for logging messages.
        """
        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._SEPARATOR)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        """
        Executes the logic to fill missing header data.

        Args:
            data (LibraryDataGenerationPipelineContext): The pipeline context containing
                                                         dataframes and configuration.

        Returns:
            LibraryDataGenerationPipelineContext: The updated pipeline context.
        """
        if not data.missing_headers:
            self._logger.debug("No headers with missing data to fill.", separator=self._SEPARATOR)
            return data

        headers_to_fill = set(data.missing_headers.keys())
        self._logger.info(f"Found {len(headers_to_fill)} headers with missing data.", separator=self._SEPARATOR)

        # --- 1. Handle Special Cases First ---
        if Header.Energy_Level in headers_to_fill:
            self._handle_energy_level_fill(data)
            headers_to_fill.remove(Header.Energy_Level)

        key_headers = {Header.Start_Key, Header.End_Key}
        if key_headers.intersection(headers_to_fill):
            self._handle_segment_keys_fill(data)
            headers_to_fill.difference_update(key_headers)

        # --- 2. Handle Standard Audio Features via Main Processor ---
        if not headers_to_fill:
            self._logger.info("All missing headers were handled by specialized processors.", separator=self._SEPARATOR)
            return data

        self._logger.info(f"Refilling {len(headers_to_fill)} standard headers via main processor.", separator=self._SEPARATOR)

        uuids_to_process = {uuid for header in headers_to_fill for uuid in data.missing_headers.get(header, [])}

        if not uuids_to_process:
            self._logger.warning("Standard headers need filling, but no associated UUIDs found.", separator=self._SEPARATOR)
            return data

        original_df = data.loaded_audio_info_cache
        df_to_process = original_df[original_df[Header.UUID.value].isin(list(uuids_to_process))]

        if df_to_process.empty:
            self._logger.info("No matching rows found in the cache for the given UUIDs.", separator=self._SEPARATOR)
            return data

        features_to_request = [HEADER_TO_FEATURE_MAPPING[header] for header in headers_to_fill]
        header_cols_to_update = [h.value for h in headers_to_fill]
        uuid_col = Header.UUID.value

        refilled_features_df = data.main_processor.process_batch(df_to_process, features_to_request)

        rename_map = {feat.name: head.value for feat, head in FEATURE_TO_HEADER_MAPPING.items() if head in headers_to_fill}
        refilled_features_df.rename(columns=rename_map, inplace=True)

        original_df[uuid_col] = original_df[uuid_col].astype(str).str.strip()
        refilled_features_df[uuid_col] = refilled_features_df[uuid_col].astype(str).str.strip()

        original_df_indexed = original_df.set_index(uuid_col)
        refilled_features_df_indexed = refilled_features_df.set_index(uuid_col)

        self._logger.debug(f"Updating columns: {header_cols_to_update}", separator=self._SEPARATOR)
        original_df_indexed.update(refilled_features_df_indexed[header_cols_to_update])

        data.loaded_audio_info_cache = original_df_indexed.reset_index()

        self._logger.info("Successfully filled standard headers and updated the cache.", separator=self._SEPARATOR)
        return data

    def _handle_energy_level_fill(self, data: LibraryDataGenerationPipelineContext):
        """Fills missing Energy Level values."""
        self._logger.info("Filling missing Energy Level...", separator=self._SEPARATOR)
        data.loaded_audio_info_cache = data.energy_calculator.calculate_ratings_for_df(
            data.loaded_audio_info_cache, Header.Energy_Level
        )
        self._logger.info("Finished filling Energy Level.", separator=self._SEPARATOR)

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

    def _handle_segment_keys_fill(self, data: LibraryDataGenerationPipelineContext):
        """Fills missing Start Key and End Key from the key progression CSV using a vectorized approach."""
        start_key_uuids = set(data.missing_headers.get(Header.Start_Key, []))
        end_key_uuids = set(data.missing_headers.get(Header.End_Key, []))
        all_uuids = start_key_uuids | end_key_uuids

        if not all_uuids:
            return

        self._logger.info("Filling missing Start/End Keys...", separator=self._SEPARATOR)
        segments_df = self._load_key_segments_df(data.key_progression_output_file_path, all_uuids)
        if segments_df is None:
            return

        # Find the first and last segments for each track efficiently
        first_segments = segments_df.loc[segments_df.groupby("Track UUID")["Segment Start"].idxmin()]
        last_segments = segments_df.loc[segments_df.groupby("Track UUID")["Segment End"].idxmax()]

        # Create a mapping from UUID to the corresponding key
        start_key_map = first_segments.set_index("Track UUID")["Segment Key"]
        end_key_map = last_segments.set_index("Track UUID")["Segment Key"]

        # Apply the updates to the main DataFrame
        df = data.loaded_audio_info_cache
        uuid_col = Header.UUID.value

        # Create boolean masks for rows that need updating
        start_mask = df[uuid_col].isin(start_key_uuids)
        end_mask = df[uuid_col].isin(end_key_uuids)

        # Use the map to apply the new keys based on UUID
        df.loc[start_mask, Header.Start_Key.value] = df.loc[start_mask, uuid_col].map(start_key_map)
        df.loc[end_mask, Header.End_Key.value] = df.loc[end_mask, uuid_col].map(end_key_map)

        self._logger.info(f"Finished filling start/end keys for {len(segments_df['Track UUID'].unique())} tracks.",
                          separator=self._SEPARATOR)
