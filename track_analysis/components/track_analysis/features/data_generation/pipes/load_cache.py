from pathlib import Path

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext


class LoadCache(IPipe):
    def __init__(self, logger: HoornLogger):
        self._separator = "BuildCSV.LoadCachePipe"
        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        self._logger.trace("Loading caches if existing...", separator=self._separator)

        main_cached_data = pd.DataFrame()
        mfcc_cached_data = pd.DataFrame()
        key_progression_cached_data = pd.DataFrame()

        if data.main_data_output_file_path.exists() and data.main_data_output_file_path.is_file():
            self._logger.trace("Main cache exists, proceeding to load...", separator=self._separator)
            main_cached_data = self._load_main_data_cache(data)

        if data.mfcc_data_output_file_path.exists() and data.mfcc_data_output_file_path.is_file():
            self._logger.trace("MFCC cache exists, proceeding to load...", separator=self._separator)
            mfcc_cached_data = self._load_mfcc_data_cache(data)

        if data.key_progression_output_file_path.exists() and data.key_progression_output_file_path.is_file():
            self._logger.trace("Key Progression cache exists, proceeding to load...", separator=self._separator)
            key_progression_cached_data = self._load_key_progression_cache(data)

        data.loaded_audio_info_cache = main_cached_data
        data.loaded_mfcc_info_cache = mfcc_cached_data
        data.loaded_key_progression_cache = key_progression_cached_data

        return data

    def _load_main_data_cache(self, data: LibraryDataGenerationPipelineContext) -> pd.DataFrame:
        return self._load_cache(data.main_data_output_file_path, "Main")

    def _load_mfcc_data_cache(self, data: LibraryDataGenerationPipelineContext) -> pd.DataFrame:
        return self._load_cache(data.mfcc_data_output_file_path, "MFCC")

    def _load_key_progression_cache(self, data: LibraryDataGenerationPipelineContext) -> pd.DataFrame:
        return self._load_cache(data.key_progression_output_file_path, "KeyProgression")

    def _load_cache(self, path: Path, descriptor: str) -> pd.DataFrame:
        cached_data: pd.DataFrame = pd.read_csv(path, header=0)
        num_records = cached_data.shape[0]

        self._logger.info(f"{descriptor} cache loaded. Number of records: {num_records}", separator=self._separator)
        return cached_data
