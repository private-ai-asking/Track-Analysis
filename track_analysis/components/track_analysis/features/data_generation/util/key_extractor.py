import ctypes
import gc
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import psutil

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import EXPENSIVE_CACHE_DIRECTORY
from track_analysis.components.track_analysis.features.key_extraction.core.config.key_progression_config import \
    KeyProgressionConfig
from track_analysis.components.track_analysis.features.key_extraction.core.definitions.definition_templates import \
    TemplateMode
from track_analysis.components.track_analysis.features.key_extraction.key_progression_analyzer import \
    KeyProgressionAnalyzer


class KeyExtractor:
    def __init__(self, logger: HoornLogger, num_workers):
        self._logger = logger
        self._separator = self.__class__.__name__

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
        self._num_workers: int = num_workers

    def extract_keys_for_track(self, track: Path, idx: int) -> Optional[Tuple[int, str, List[Tuple[float, float, str]]]]:
        try:
            local_keys, global_key = self._key_extractor.analyze(track)
            global_key = self._key_extractor.convert_label_to_camelot(global_key)

            if local_keys is None or global_key is None:
                self._logger.error(f"There was an error extracting keys from: \"{track}\"!", separator=self._separator)
                return None

            local_info: List[Tuple[float, float, str]] = []
            for local_key in local_keys:
                camelot_label = self._key_extractor.convert_label_to_camelot(local_key.state_label)
                local_info.append((local_key.start_time, local_key.end_time, camelot_label))

            return idx, global_key, local_info
        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error(f"Something went wrong during local key analysis for track \"{track}\"!\n{e}\n{tb}", separator=self._separator)
            return None

    def extract_keys_batch(
            self, tracks: List[Tuple[int, Path]]
    ) -> List[Tuple[int, str, List[Tuple[float, float, str]]]]:
        """
        Process the list of (idx, path) pairs in batches, extract global and segment keys,
        then return a list of tuples: (index, global_key, [(segment_start, segment_end, segment_key), ...]).
        """
        batch_size = 50
        final_results: List[Tuple[int, str, List[Tuple[float, float, str]]]] = []

        for start in range(0, len(tracks), batch_size):
            batch_items = tracks[start : start + batch_size]
            batch_results = self._process_batch(batch_items)
            final_results.extend(batch_results)

            self._logger.info(
                f"Completed batch of {len(batch_items)} tracks. Running cleanup.",
                separator=self._separator,
            )
            self._trim_and_defrag(
                after_description=f"after batch starting at index {start}"
            )

        return final_results

    def _process_batch(
            self, batch_items: List[Tuple[int, Path]]
    ) -> List[Tuple[int, str, List[Tuple[float, float, str]]]]:
        """
        Submit each track in the batch to extract_keys_for_track in parallel.
        Collect and aggregate results per track index.
        """
        batch_results: Dict[int, Tuple[str, List[Tuple[float, float, str]]]] = {}

        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            futures = {
                executor.submit(self.extract_keys_for_track, path, idx): idx
                for idx, path in batch_items
            }

            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue

                idx, global_key, local_info = result
                segments: List[Tuple[float, float, str]] = [
                    (start, end, key) for start, end, key in local_info
                ]
                batch_results[idx] = (global_key, segments)

        # Convert the batch_results dict into the desired list-of-tuples format
        return [
            (idx, global_key, segments)
            for idx, (global_key, segments) in batch_results.items()
        ]

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
