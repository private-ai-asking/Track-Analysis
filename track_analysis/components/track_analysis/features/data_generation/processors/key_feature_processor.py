from typing import List

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.helpers.key_extractor import KeyExtractor, \
    KeyExtractionResult


class KeyFeatureProcessor:
    def __init__(self, key_extractor: KeyExtractor, logger: HoornLogger):
        self._key_extractor = key_extractor
        self._logger = logger
        self._separator = "BuildCSV.KeyFeatureProcessor"

    def extract_raw_keys(self, meta_df: pd.DataFrame) -> List[KeyExtractionResult]:
        """
        Runs key extraction for a batch and returns the raw result objects.
        The method name is changed to be more descriptive of its focused task.
        """
        self._logger.info("Extracting raw key data for batch...", separator=self._separator)

        indexed_paths = list(meta_df[[Header.Audio_Path.value]].itertuples(name=None))
        key_extraction_results = self._key_extractor.extract_keys_batch(indexed_paths)

        return key_extraction_results
