from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
import traceback

import mutagen
import numpy as np
import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.util.key_extractor import (
    KeyExtractor,
    KeyExtractionResult,
)

# A more explicit data structure for the final output.
@dataclass
class KeyTaggingResults:
    """
    Holds the results of the key tagging and analysis process.
    """
    global_keys: np.ndarray
    start_keys: np.ndarray
    end_keys: np.ndarray
    key_extraction_results: List[KeyExtractionResult]
    key_progression_df: pd.DataFrame

@dataclass
class KeyProgressionRow:
    """
    Represents a single row for the key progression DataFrame.
    """
    file_path: str
    segment_start: float
    segment_end: float
    segment_key: str

class KeyTaggingService:
    def __init__(self, extractor: KeyExtractor, csv_path: Path, logger: HoornLogger):
        self._extractor = extractor
        self._csv_path = csv_path
        self._logger = logger
        self._separator = self.__class__.__name__

    def tag_and_record(self, paths: List[Path]) -> KeyTaggingResults:
        """
        Extracts keys from audio files, tags the files with global keys, and
        generates a DataFrame of key progression for each file.

        Args:
            paths: A list of Path objects to the audio files.

        Returns:
            A KeyTaggingResults object containing the comprehensive output.
        """
        extraction_results = self._safe_extract_keys(paths)

        # Initialize output arrays and lists
        num_files = len(paths)
        global_keys = np.full(num_files, "", dtype=object)
        start_keys = np.full(num_files, "", dtype=object)
        end_keys = np.full(num_files, "", dtype=object)
        key_progression_rows: List[KeyProgressionRow] = []

        # Process each extraction result
        for res in extraction_results:
            self._process_single_result(
                res,
                paths,
                global_keys,
                start_keys,
                end_keys,
                key_progression_rows
            )

        # Create the final DataFrame
        key_progression_df = (
            pd.DataFrame([row.__dict__ for row in key_progression_rows])
            if key_progression_rows
            else pd.DataFrame()
        )

        return KeyTaggingResults(
            global_keys=global_keys,
            start_keys=start_keys,
            end_keys=end_keys,
            key_extraction_results=extraction_results,
            key_progression_df=key_progression_df,
        )

    def _safe_extract_keys(self, paths: List[Path]) -> List[KeyExtractionResult]:
        """
        Safely calls the key extraction service and handles exceptions.
        """
        try:
            return self._extractor.extract_keys_batch(list(enumerate(paths)))
        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error(f"KeyExtractor failed: {e}\n{tb}", separator=self._separator)
            return []

    def _process_single_result(
            self,
            result: KeyExtractionResult,
            paths: List[Path],
            global_keys: np.ndarray,
            start_keys: np.ndarray,
            end_keys: np.ndarray,
            key_progression_rows: List[KeyProgressionRow]
    ) -> None:
        """
        Processes a single key extraction result, updating the main output arrays
        and the list of rows for the key progression DataFrame.
        """
        idx = result.index
        path = paths[idx]

        # Update global key array
        global_keys[idx] = result.global_key

        # Tag FLAC files
        if path.suffix.lower() == ".flac":
            self._write_flac_key_tag(path, result.global_key)

        # Process local key information if available
        if result.local_info:
            # Update start and end key arrays
            start_keys[idx] = result.local_info[0].key
            end_keys[idx] = result.local_info[-1].key

            # Append rows for the key progression DataFrame
            for segment in result.local_info:
                key_progression_rows.append(
                    KeyProgressionRow(
                        file_path=str(path),
                        segment_start=segment.interval_start,
                        segment_end=segment.interval_end,
                        segment_key=segment.key,
                    )
                )

    def _write_flac_key_tag(self, path: Path, key: str) -> None:
        """
        Writes the 'initialkey' and 'global_key' tags to a FLAC file.
        Failures are logged but do not stop the process.
        """
        try:
            tag_file = mutagen.File(str(path), easy=True)
            if tag_file is None:
                return
            tag_file["initialkey"] = key
            tag_file["global_key"] = key
            tag_file.save()
        except Exception as e:
            # Best effort; failures here shouldn't stop the pipeline
            self._logger.warning(
                f"Failed to write key tag to {path}: {e}",
                separator=self._separator
            )
            pass
