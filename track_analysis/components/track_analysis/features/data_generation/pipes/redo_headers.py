import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.feature_to_header_mapping import \
    HEADER_TO_FEATURE_MAPPING, FEATURE_TO_HEADER_MAPPING
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext


class RedoHeaders(IPipe):
    """
    Orchestrator class that delegates header processing to specialized processor classes.
    """

    _SEPARATOR = "BuildCSV.RedoHeaders"

    def __init__(self, logger: HoornLogger):
        self._logger = logger

        self._logger.trace("Successfully initialized pipe.", separator=self._SEPARATOR)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        if Header.Energy_Level in data.headers_to_refill:
            self._handle_energy_level_refill(data)
            data.headers_to_refill.remove(Header.Energy_Level)

        if not data.headers_to_refill:
            self._logger.debug("No headers to refill.", separator=self._SEPARATOR)
            return data

        self._logger.info(f"Refilling {len(data.headers_to_refill)} headers.", separator=self._SEPARATOR)

        features_to_request = [HEADER_TO_FEATURE_MAPPING[header] for header in data.headers_to_refill]
        header_cols_to_update = [h.value for h in data.headers_to_refill]
        audio_unique_identifier_col = Header.UUID.value

        original_df = data.loaded_audio_info_cache
        df_to_process = original_df
        refilled_features_df = data.main_processor.process_batch(df_to_process, features_to_request)

        rename_map = {
            feature.name: header.value
            for feature, header in FEATURE_TO_HEADER_MAPPING.items()
        }

        refilled_features_df.rename(columns=rename_map, inplace=True)

        for col in header_cols_to_update:
            if col in original_df.columns and col in refilled_features_df.columns:
                original_df[col] = pd.to_numeric(original_df[col], errors='coerce')
                refilled_features_df[col] = pd.to_numeric(refilled_features_df[col], errors='coerce')

        original_df[audio_unique_identifier_col] = original_df[audio_unique_identifier_col].astype(str).str.strip()
        refilled_features_df[audio_unique_identifier_col] = refilled_features_df[audio_unique_identifier_col].astype(str).str.strip()

        original_df_indexed = original_df.set_index(audio_unique_identifier_col)
        refilled_features_df_indexed = refilled_features_df.set_index(audio_unique_identifier_col)

        original_df_indexed.update(refilled_features_df_indexed[header_cols_to_update])

        data.loaded_audio_info_cache = original_df_indexed.reset_index()

        self._logger.info("Successfully refilled headers and updated the cache.", separator=self._SEPARATOR)
        return data

    def _handle_energy_level_refill(self, data: LibraryDataGenerationPipelineContext):
        self._logger.info("Refilling energy levels...", separator=self._SEPARATOR)
        data.loaded_audio_info_cache = data.energy_calculator.calculate_ratings_for_df(data.loaded_audio_info_cache, Header.Energy_Level)
