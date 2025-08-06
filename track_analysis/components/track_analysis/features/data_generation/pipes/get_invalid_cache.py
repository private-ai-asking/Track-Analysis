from pathlib import Path
from typing import List, Set

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import LibraryDataGenerationPipelineContext


class HandleInvalidCache(IPipe):
    _SEPARATOR = "BuildCSV.GetInvalidCachePipe"

    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._SEPARATOR)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        self._logger.trace("Validating cache entries...", separator=self._SEPARATOR)

        if not data.loaded_audio_info_cache.empty:
            self._validate_audio_cache(data)

        if not data.loaded_mfcc_info_cache.empty:
            self._validate_mfcc_cache(data)

        if not data.loaded_key_progression_cache.empty:
            self._validate_key_progression_cache(data)

        self._logger.info("Cache validation complete.", separator=self._SEPARATOR)
        return data

    def _validate_audio_cache(self, data: LibraryDataGenerationPipelineContext):
        """
        Validates the audio info cache for duplicates and invalid paths.
        """
        audio_paths: pd.Series = data.loaded_audio_info_cache[Header.Audio_Path.value]

        self._log_duplicates(audio_paths)
        self._log_and_set_invalid_paths(audio_paths, data)

    def _log_duplicates(self, audio_paths: pd.Series):
        """
        Logs any duplicate audio paths found in the cache.
        """
        dup_counts = audio_paths.value_counts()
        duplicates = dup_counts[dup_counts > 1].index.tolist()

        if duplicates:
            self._logger.warning(
                f"Duplicate cache entries detected for {len(duplicates)} paths: {duplicates[:10]}",
                separator=self._SEPARATOR,
            )

    def _log_and_set_invalid_paths(self, audio_paths: pd.Series, data: LibraryDataGenerationPipelineContext):
        """
        Checks for invalid paths (non-existent files) and logs the results.
        """
        unique_paths: Set[str] = set(audio_paths)
        invalid_paths: List[Path] = [
            Path(path_str) for path_str in unique_paths
            if not Path(path_str).is_file()
        ]

        total_cached = len(audio_paths)
        if total_cached > 0:
            self._logger.info(
                f"Found invalid cached paths: {len(invalid_paths)}/{total_cached} "
                f"({round(len(invalid_paths) / total_cached * 100, 2)}%)",
                separator=self._SEPARATOR,
            )

        data.invalid_cached_paths = invalid_paths

    def _validate_mfcc_cache(self, data: LibraryDataGenerationPipelineContext):
        """
        Validates the MFCC cache by ensuring all entries have a corresponding audio entry.
        """
        data.loaded_mfcc_info_cache = self._validate_cache(data, data.loaded_mfcc_info_cache, "MFCC")

    def _validate_key_progression_cache(self, data: LibraryDataGenerationPipelineContext):
        """
        Validates the Key Progression cache by ensuring all entries have a corresponding audio entry.
        """
        data.loaded_key_progression_cache = self._validate_cache(data, data.loaded_key_progression_cache, "Key Progression", uuid_header_other_df="Track UUID")

    def _validate_cache(self, data: LibraryDataGenerationPipelineContext, df: pd.DataFrame, descriptor: str, uuid_header_other_df: str = Header.UUID.value) -> pd.DataFrame:
        existing_uuids: pd.Series = data.loaded_audio_info_cache[Header.UUID.value]

        is_valid = df[uuid_header_other_df].isin(existing_uuids)

        invalid_count = (~is_valid).sum()
        if invalid_count > 0:
            self._logger.warning(
                f"Found {invalid_count} invalid {descriptor} entries. Cleaning cache.",
                separator=self._SEPARATOR
            )
            df = df[is_valid]

        return df
