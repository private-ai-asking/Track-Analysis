import ctypes
import gc
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Callable, Tuple, Optional

import mutagen
import numpy as np
import pandas as pd
import psutil

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.constants import EXPENSIVE_CACHE_DIRECTORY
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler, AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.key_extraction.core.config.key_progression_config import \
    KeyProgressionConfig
from track_analysis.components.track_analysis.features.key_extraction.core.definitions.definition_templates import \
    TemplateMode
from track_analysis.components.track_analysis.features.key_extraction.key_progression_analyzer import \
    KeyProgressionAnalyzer


class RedoHeaders(IPipe):
    def __init__(self, logger: HoornLogger, file_handler: AudioFileHandler, num_workers: int):
        self._separator = "BuildCSV.RedoHeaders"
        self._file_handler = file_handler
        self._num_workers = num_workers

        self._header_processor_func_mapping: Dict[Header, Callable[[List[str], LibraryDataGenerationPipelineContext], None]] = {
            Header.BPM: self._redo_bpm,
            Header.Key: self._redo_key
        }

        self._logger = logger

        config: KeyProgressionConfig = KeyProgressionConfig(
            tone_modulation_penalty=18,
            mode_modulation_penalty=None,
            visualize=False,
            template_mode=TemplateMode.HOORN,
            hop_length=512,
            subdivisions_per_beat=2,
            segment_beat_level=4,
            cache_dir=EXPENSIVE_CACHE_DIRECTORY
        )

        self._key_extractor: KeyProgressionAnalyzer = KeyProgressionAnalyzer(logger, config)
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        for header, uuids in data.refill_headers.items():
            processor = self._header_processor_func_mapping.get(header, None)
            if processor is None:
                self._logger.warning(f"Header {header} not found, unsupported refill... skipping.", separator=self._separator)
                continue
            processor(uuids, data)

        return data

    def _redo_key(self, uuids: List[str], data: LibraryDataGenerationPipelineContext):
        df: pd.DataFrame = data.loaded_audio_info_cache

        # 1) Filter out only those rows whose UUID is in our target list
        rows: pd.DataFrame = df.loc[df[Header.UUID.value].isin(uuids)]

        # 2) Build a list of (original_df_index, Path) tuples for ALL matching rows
        all_items: List[Tuple[int, Path]] = [
            (idx, Path(p))
            for idx, p in zip(rows.index, rows[Header.Audio_Path.value].tolist())
        ]

        # 3) We will process in batches of size `data.max_new_tracks_per_run`
        batch_size = data.max_new_tracks_per_run
        total_items = len(all_items)
        processed_count = 0

        local_keys_storage: List[Tuple[str, float, float, str]] = []

        # 4) Loop over all_items in chunks
        for start in range(0, total_items, batch_size):
            end = start + batch_size
            batch_items = all_items[start:end]

            # Define a worker function to run in threads
            def _process_item(idx: int, path: Path) -> Optional[Tuple[int, str, List[Tuple[float, float, str]]]]:
                """
                Analyze `path`, write tags if needed, and return:
                  - idx (to update df)
                  - global_key (converted to Camelot)
                  - list of (segment_start, segment_end, segment_key) tuples
                """
                try:
                    # 5) Analyze file for local and global keys
                    local_keys, global_key = self._key_extractor.analyze(path)

                    if local_keys is None or global_key is None:
                        self._logger.error(f"There was an error extracting keys from: \"{path}\"!", separator=self._separator)
                        return None

                    global_key = self._key_extractor.convert_label_to_camelot(global_key)

                    # 6) If it's a .flac file, write the 'bpm' tag to disk
                    if path.suffix.lower() == ".flac":
                        tag_file: mutagen.File = mutagen.File(str(path), easy=True)
                        tag_file["initialkey"] = global_key
                        tag_file["global_key"] = global_key
                        tag_file.save()

                    # 7) Build a list of converted local-key info
                    local_info: List[Tuple[float, float, str]] = []
                    for local_key in local_keys:
                        camelot_label = self._key_extractor.convert_label_to_camelot(local_key.state_label)
                        local_info.append((local_key.start_time, local_key.end_time, camelot_label))

                    # 8) Return everything needed for the main thread to update df and storage
                    return idx, global_key, local_info
                except Exception as e:
                    tb = traceback.format_exc()
                    self._logger.error(f"Something went wrong during local key analysis for track \"{path}\"!\n{e}\n{tb}", separator=self._separator)
                    return None

            # 9) Use ThreadPoolExecutor to process this batch in parallel
            with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                # Submit one future per (idx, path)
                futures = {
                    executor.submit(_process_item, idx, path): idx
                    for idx, path in batch_items
                }

                # As each future completes, update df and local_keys_storage
                for future in as_completed(futures):
                    result = future.result()

                    if result is None:
                        continue

                    idx = result[0]
                    global_key = result[1]
                    local_info = result[2]

                    uuid: str = uuids[idx]

                    # Append each segment's key info to local_keys_storage
                    for segment_start, segment_end, segment_key in local_info:
                        local_keys_storage.append((uuid, segment_start, segment_end, segment_key))

                    # Update the DataFrame with the global key
                    df.loc[idx, Header.Key.value] = global_key

            processed_count += len(batch_items)

            # 10) Cleanup memory after each batch
            self._logger.info(
                f"Completed batch of {len(batch_items)} tracks. Running cleanup.",
                separator=self._separator,
            )
            self._trim_and_defrag(after_description=f"after batch starting at index {start}")

        # 11) After all batches: write out local_keys_storage to CSV
        local_keys_df = pd.DataFrame(
            local_keys_storage,
            columns=["Track UUID", "Segment Start", "Segment End", "Segment Key"],
        )
        local_keys_df.to_csv(data.key_progression_output_file_path, index=False)

        # 12) Final log once all batches are done
        self._logger.info(
            f"Finished processing {processed_count} tracks with redoing BPM values!",
            separator=self._separator,
        )

    def _redo_bpm(self, uuids: List[str], data: LibraryDataGenerationPipelineContext) -> None:
        df: pd.DataFrame = data.loaded_audio_info_cache

        # 1) Filter out only those rows whose UUID is in our target list
        rows: pd.DataFrame = df.loc[df[Header.UUID.value].isin(uuids)]

        # 2) Build a list of (original_df_index, Path) tuples for ALL matching rows
        all_items: List[Tuple[int, Path]] = [
            (idx, Path(p))
            for idx, p in zip(rows.index, rows[Header.Audio_Path.value].tolist())
        ]

        # 3) We will process in batches of size `data.max_new_tracks_per_run`
        batch_size = data.max_new_tracks_per_run
        total_items = len(all_items)
        processed_count = 0

        # 4) Loop over all_items in chunks
        for start in range(0, total_items, batch_size):
            end = start + batch_size
            batch_items = all_items[start:end]

            # Unzip indices and paths for this batch
            batch_indices = [idx for idx, _ in batch_items]
            batch_paths = [path for _, path in batch_items]

            # 5) Fetch BPMs for this batch
            audio_infos: List[AudioStreamsInfoModel] = self._file_handler.get_audio_streams_info_batch(batch_paths)
            bpms: List[float] = [info.tempo for info in audio_infos]

            # 6) Write BPM back into the DataFrame at the correct indices
            df.loc[batch_indices, Header.BPM.value] = bpms

            # 7) Also write the 'bpm' tag into each file on disk
            for bpm, path in zip(bpms, batch_paths):
                self._logger.debug(f"Detected BPM: {bpm:.4f} for \"{path}\"", separator=self._separator)
                tag_file: mutagen.File = mutagen.File(str(path), easy=True)
                tag_file["bpm"] = str(round(bpm, 4))
                tag_file.save()

            processed_count += len(batch_items)

            # 8) Cleanup memory after each batch
            self._logger.info(f"Completed batch of {len(batch_items)} tracks. Running cleanup.", separator=self._separator)
            self._trim_and_defrag(after_description=f"after batch starting at index {start}")

        # 9) Final log once all batches are done
        self._logger.info(
            f"Finished processing {processed_count} tracks with redoing BPM values!",
            separator=self._separator
        )

    def _trim_and_defrag(self, after_description: str = ""):
        """
        Free Python-level objects, force GC, trim Windows working set, touch scratch.
        """
        # Force Python GC
        gc.collect()

        # Trim Windows working set (if on Windows)
        try:
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
        except Exception:
            pass

        # Scratch allocate ~200 MiB of float32 to pin contiguous pages
        try:
            scratch = np.empty((52_000_000,), dtype=np.float32)
            scratch.fill(0)
            del scratch
            gc.collect()
        except Exception:
            pass

        # Print available memory for debugging
        vm = psutil.virtual_memory()
        proc = psutil.Process()
        self._logger.debug(f"[{after_description}] Total RAM:   {vm.total >> 20} MiB", separator=self._separator)
        self._logger.debug(f"[{after_description}] Available:   {vm.available >> 20} MiB", separator=self._separator)
        self._logger.debug(f"[{after_description}] Process RSS: {proc.memory_info().rss >> 20} MiB", separator=self._separator)
