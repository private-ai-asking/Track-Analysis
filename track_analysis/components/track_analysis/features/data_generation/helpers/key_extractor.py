import ctypes
import dataclasses
import gc
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import psutil

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import EXPENSIVE_CACHE_DIRECTORY
from track_analysis.components.track_analysis.legacy.audio_file_handler import AudioFileHandler
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.core.config.key_progression_config import \
    KeyProgressionConfig
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.core.definitions.definition_templates import \
    TemplateMode
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.key_progression_analyzer import \
    KeyProgressionAnalyzer
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.preprocessing.note_extraction.notes.note_event_builder import \
    NoteEvent


@dataclasses.dataclass
class LocalKeyProgressionInfo:
    interval_start: float
    interval_end: float
    key: str

@dataclasses.dataclass
class KeyExtractionResult:
    index: int
    global_key: str
    local_info: List[LocalKeyProgressionInfo]
    note_events: List[NoteEvent]


class KeyExtractor:
    def __init__(
            self,
            logger: HoornLogger,
            audio_loader: AudioFileHandler,
            num_workers: int
    ):
        self._logger = logger
        self._separator = self.__class__.__name__

        config = KeyProgressionConfig(
            tone_modulation_penalty=18,
            mode_modulation_penalty=None,
            visualize=False,
            template_mode=TemplateMode.HOORN,
            hop_length=512,
            subdivisions_per_beat=2,
            segment_beat_level=4,
            cache_dir=EXPENSIVE_CACHE_DIRECTORY,
        )

        self._analyzer = KeyProgressionAnalyzer(logger, config, audio_loader)
        self._num_workers = num_workers

    def extract_keys_for_track(
            self,
            idx: int,
            track: Path
    ) -> Optional[KeyExtractionResult]:
        try:
            local_keys, global_key_label, track_notes = self._analyzer.analyze(track)
            if not local_keys or not global_key_label:
                self._logger.error(
                    f"Key extraction failed for track: {track}",
                    separator=self._separator
                )
                return None

            global_key = self._analyzer.convert_label_to_camelot(global_key_label)

            local_info = [
                LocalKeyProgressionInfo(
                    interval_start=lk.start_time,
                    interval_end=lk.end_time,
                    key=self._analyzer.convert_label_to_camelot(lk.state_label)
                )
                for lk in local_keys
            ]

            return KeyExtractionResult(
                index=idx,
                global_key=global_key,
                local_info=local_info,
                note_events=track_notes,
            )

        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error(
                f"Error during key extraction for {track}: {e}\n{tb}",
                separator=self._separator
            )
            return None

    def extract_keys_batch(
            self,
            tracks: List[Tuple[int, Path]]
    ) -> List[KeyExtractionResult]:
        """
        Parallel extraction of key info for a batch of (index, track) pairs.
        Returns a list of KeyExtractionResult.
        """
        results: List[KeyExtractionResult] = []

        # process in fixed-size batches to control memory
        batch_size = 50
        for start in range(0, len(tracks), batch_size):
            slice_ = tracks[start : start + batch_size]
            batch_results = self._process_batch(slice_)
            results.extend(batch_results)

            self._logger.info(
                f"Completed key extraction for batch {start}-{start+len(slice_)}.",
                separator=self._separator,
            )
            self._trim_and_defrag(f"after batch {start}")

        return results

    def _process_batch(
            self,
            batch_items: List[Tuple[int, Path]]
    ) -> List[KeyExtractionResult]:
        """
        Uses a ThreadPoolExecutor to parallelize per-track key extraction.
        """
        results: List[KeyExtractionResult] = []
        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            future_to_idx: Dict = {
                executor.submit(self.extract_keys_for_track, idx, path): idx
                for idx, path in batch_items
            }
            for future in as_completed(future_to_idx):
                res = future.result()
                if res:
                    results.append(res)
        return results

    def _trim_and_defrag(self, after_description: str = ""):
        """
        Releases Python objects, forces GC, trims working set on Windows,
        and touches a scratch buffer to defragment.
        """
        gc.collect()
        try:
            ctypes.windll.kernel32.SetProcessWorkingSetSize(
                -1,
                ctypes.c_size_t(-1),
                ctypes.c_size_t(-1)
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
            f"[{after_description}] RAM total: {vm.total >> 20} MiB, "
            f"available: {vm.available >> 20} MiB, "
            f"RSS: {proc.memory_info().rss >> 20} MiB",
            separator=self._separator
        )
