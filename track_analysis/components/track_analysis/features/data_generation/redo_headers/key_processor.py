from pathlib import Path
from typing import List, Tuple

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler
from track_analysis.components.track_analysis.features.core.memory_utils import trim_and_defrag
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.data_generation.redo_headers.base_header_processor import \
    BaseHeaderProcessor
from track_analysis.components.track_analysis.features.data_generation.util.key_extractor import KeyExtractor


class KeyProcessor(BaseHeaderProcessor):
    def __init__(self, logger: HoornLogger, file_handler: AudioFileHandler, key_extractor: KeyExtractor):
        super().__init__(logger, file_handler)
        self._key_extractor = key_extractor

    def process(self, uuids: List[str], data: LibraryDataGenerationPipelineContext) -> None:
        df = data.loaded_audio_info_cache
        items = self._collect_index_path_pairs(df, uuids)
        processed_count = 0
        local_keys_storage = []

        extractor_results = self._log_and_handle_exception(
            "KeyExtractor.extract_keys_batch failed",
            self._key_extractor.extract_keys_batch,
            items
        )

        if not extractor_results:
            self._logger.warning("No key extraction results to process.", separator=self._SEPARATOR)
            return

        for idx, global_key, segments in extractor_results:
            if global_key:
                self._update_dataframe_column(df, [idx], Header.Key, [global_key])
                self._write_flac_key_tag(df, idx, global_key)
                self._collect_local_keys(df, idx, segments, local_keys_storage)
                processed_count += 1

                if processed_count % data.max_new_tracks_per_run == 0:
                    self._logger.info(
                        f"Processed {processed_count} key extractions so far. Running cleanup.",
                        separator=self._SEPARATOR
                    )
                    trim_and_defrag(self._logger, f"after processing {processed_count} tracks")

        self._write_local_keys_csv(local_keys_storage, data.key_progression_output_file_path)
        self._logger.info(
            f"Finished processing {processed_count} tracks for key extraction!",
            separator=self._SEPARATOR
        )

    def _write_flac_key_tag(self, df: pd.DataFrame, idx: int, key: str) -> None:
        path = Path(df.loc[idx, Header.Audio_Path.value])
        if path.suffix.lower() == ".flac":
            self._update_audio_tags([path, path], "initialkey", [key, key])

    def _collect_local_keys(self, df: pd.DataFrame, idx: int, segments: List[Tuple[float, float, str]], storage: List) -> None:
        self._log_and_handle_exception(
            f"Error collecting local keys for idx {idx}",
            lambda: self._append_local_keys_to_storage(df, idx, segments, storage)
        )

    def _append_local_keys_to_storage(self, df, idx, segments, storage):
        track_uuid = df.loc[idx, Header.UUID.value]
        for start, end, raw_label in segments:
            storage.append((track_uuid, start, end, raw_label))

    def _write_local_keys_csv(self, local_keys: List, output_path: Path) -> None:
        self._log_and_handle_exception(
            f"Failed to write local keys CSV to {output_path}",
            lambda: pd.DataFrame(
                local_keys,
                columns=["Track UUID", "Segment Start", "Segment End", "Segment Key"]
            ).to_csv(output_path, index=False)
        )
