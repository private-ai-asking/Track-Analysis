import ctypes
import gc
import traceback
from pathlib import Path
from typing import List, Dict, Callable, Tuple

import mutagen
import numpy as np
import pandas as pd
import psutil

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler, AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.data_generation.util.key_extractor import KeyExtractor


class RedoHeaders(IPipe):
    def __init__(self, logger: HoornLogger, file_handler: AudioFileHandler, num_workers: int):
        self._separator = "BuildCSV.RedoHeaders"
        self._logger = logger
        self._file_handler = file_handler
        self._num_workers = num_workers

        self._key_extractor: KeyExtractor = KeyExtractor(logger, file_handler, num_workers)

        self._header_processor: Dict[Header, Callable[[List[str], LibraryDataGenerationPipelineContext], None]] = {
            Header.BPM: self._redo_bpm,
            Header.Key: self._redo_key
        }

        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        for header, uuids in data.refill_headers.items():
            processor = self._header_processor.get(header)
            if not processor:
                self._logger.warning(
                    f"Header {header} not found, unsupported refill... skipping.",
                    separator=self._separator
                )
                continue
            processor(uuids, data)
        return data

    # ----------------------- Key Refill -----------------------

    def _redo_key(self, uuids: List[str], data: LibraryDataGenerationPipelineContext) -> None:
        df = data.loaded_audio_info_cache
        items = self._collect_index_path_pairs(df, uuids)
        extractor_results = self._run_key_extraction(items)
        local_keys = []
        processed = 0

        for idx, global_key, segments in extractor_results:
            success = self._update_dataframe_key(df, idx, global_key)
            if success:
                self._write_flac_key_tag(df, idx, global_key)
                self._collect_local_keys(df, idx, segments, local_keys)
                processed += 1

            if processed % data.max_new_tracks_per_run == 0:
                self._logger.info(
                    f"Processed {processed} key extractions so far. Running cleanup.",
                    separator=self._separator
                )
                self._trim_and_defrag(after_description=f"after processing {processed} tracks")

        self._write_local_keys_csv(local_keys, data.key_progression_output_file_path)
        self._logger.info(
            f"Finished processing {processed} tracks for key extraction!",
            separator=self._separator
        )

    @staticmethod
    def _collect_index_path_pairs(df: pd.DataFrame, uuids: List[str]) -> List[Tuple[int, Path]]:
        rows = df.loc[df[Header.UUID.value].isin(uuids)]
        return [
            (idx, Path(p))
            for idx, p in zip(rows.index, rows[Header.Audio_Path.value].tolist())
        ]

    def _run_key_extraction(
            self, items: List[Tuple[int, Path]]
    ) -> List[Tuple[int, str, List[Tuple[float, float, str]]]]:
        try:
            return self._key_extractor.extract_keys_batch(items)
        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error(
                f"KeyExtractor.extract_keys_batch failed: {e}\n{tb}",
                separator=self._separator
            )
            return []

    def _update_dataframe_key(self, df: pd.DataFrame, idx: int, key: str) -> bool:
        try:
            df.loc[idx, Header.Key.value] = key
            return True
        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error(
                f"Error updating DataFrame for idx {idx}: {e}\n{tb}",
                separator=self._separator
            )
            return False

    def _write_flac_key_tag(self, df: pd.DataFrame, idx: int, key: str) -> None:
        path = Path(df.loc[idx, Header.Audio_Path.value])
        if path.suffix.lower() != ".flac":
            return
        try:
            tag_file = mutagen.File(str(path), easy=True)
            tag_file["initialkey"] = key
            tag_file["global_key"] = key
            tag_file.save()
        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error(
                f"Failed to write FLAC key tag for idx {idx}, path {path}: {e}\n{tb}",
                separator=self._separator
            )

    def _collect_local_keys(
            self,
            df: pd.DataFrame,
            idx: int,
            segments: List[Tuple[float, float, str]],
            storage: List[Tuple[str, float, float, str]]
    ) -> None:
        try:
            track_uuid = df.loc[idx, Header.UUID.value]
            for start, end, raw_label in segments:
                storage.append((track_uuid, start, end, raw_label))
        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error(
                f"Error collecting local keys for idx {idx}: {e}\n{tb}",
                separator=self._separator
            )

    def _write_local_keys_csv(
            self,
            local_keys: List[Tuple[str, float, float, str]],
            output_path: Path
    ) -> None:
        try:
            df_local = pd.DataFrame(
                local_keys,
                columns=["Track UUID", "Segment Start", "Segment End", "Segment Key"]
            )
            df_local.to_csv(output_path, index=False)
        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error(
                f"Failed to write local keys CSV to {output_path}: {e}\n{tb}",
                separator=self._separator
            )

    # ----------------------- BPM Refill -----------------------

    def _redo_bpm(self, uuids: List[str], data: LibraryDataGenerationPipelineContext) -> None:
        df = data.loaded_audio_info_cache
        items = self._collect_index_path_pairs(df, uuids)
        batch_size = data.max_new_tracks_per_run
        processed = 0

        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            indices, paths = zip(*batch)
            bpms = self._fetch_bpms(list(paths))
            self._update_dataframe_bpm(df, list(indices), bpms)
            self._write_bpm_tags(list(paths), bpms)
            processed += len(batch)

            self._logger.info(
                f"Completed batch of {len(batch)} BPM updates. Running cleanup.",
                separator=self._separator
            )
            self._trim_and_defrag(after_description=f"after BPM batch starting at index {start}")

        self._logger.info(
            f"Finished processing {processed} tracks with redoing BPM values!",
            separator=self._separator
        )

    def _fetch_bpms(self, paths: List[Path]) -> List[float]:
        try:
            infos: List[AudioStreamsInfoModel] = self._file_handler.get_audio_streams_info_batch(paths)
            return [info.tempo for info in infos]
        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error(
                f"Failed to fetch BPMs for batch: {e}\n{tb}",
                separator=self._separator
            )
            return [0.0] * len(paths)

    def _update_dataframe_bpm(self, df: pd.DataFrame, indices: List[int], bpms: List[float]) -> None:
        try:
            df.loc[indices, Header.BPM.value] = bpms
        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error(
                f"Error updating DataFrame BPM values for indices {indices}: {e}\n{tb}",
                separator=self._separator
            )

    def _write_bpm_tags(self, paths: List[Path], bpms: List[float]) -> None:
        for path, bpm in zip(paths, bpms):
            try:
                self._logger.debug(
                    f"Detected BPM: {bpm:.4f} for \"{path}\"",
                    separator=self._separator
                )
                tag_file = mutagen.File(str(path), easy=True)
                tag_file["bpm"] = str(round(bpm, 4))
                tag_file.save()
            except Exception as e:
                tb = traceback.format_exc()
                self._logger.error(
                    f"Failed to write BPM tag for \"{path}\": {e}\n{tb}",
                    separator=self._separator
                )

    # ----------------------- Memory Cleanup -----------------------

    def _trim_and_defrag(self, after_description: str = ""):
        gc.collect()
        try:
            ctypes.windll.kernel32.SetProcessWorkingSetSize(
                -1, ctypes.c_size_t(-1), ctypes.c_size_t(-1)
            )
        except Exception:
            pass

        try:
            scratch = np.empty((52_000_000,), dtype=np.float32)
            scratch.fill(0)
            del scratch
            gc.collect()
        except Exception:
            pass

        vm = psutil.virtual_memory()
        proc = psutil.Process()
        self._logger.debug(
            f"[{after_description}] Total RAM:   {vm.total >> 20} MiB",
            separator=self._separator
        )
        self._logger.debug(
            f"[{after_description}] Available:   {vm.available >> 20} MiB",
            separator=self._separator
        )
        self._logger.debug(
            f"[{after_description}] Process RSS: {proc.memory_info().rss >> 20} MiB",
            separator=self._separator
        )
