import math
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
from pyebur128.pyebur128 import R128State, MeasurementMode, get_loudness_range, get_loudness_global, get_true_peak

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import CACHE_DIRECTORY
from track_analysis.components.track_analysis.features.audio_file_handler import AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class AudioCalculator:
    def __init__(self, logger: HoornLogger):
        self._separator = "AudioCalculator"
        self._logger = logger

        self._max_data_rate_cache_path: Path = CACHE_DIRECTORY / "max_data_rate_cache.pkl"
        self._max_data_rate_cache_lookup: Dict[Tuple[float, int, int], float] = self._load_max_data_rate_cache_lookup()

        self._processed: int = 0

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def calculate_batch_sample_metrics(
            self,
            samples_list: List[np.ndarray],
            sample_rates: List[float],
            chunk_size: int = 4096,
            max_workers: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Single‐pass batch calculation for:
          - True-Peak (dBTP)
          - Integrated Loudness (LUFS)
          - Loudness Range (LRA, in dB)
          - Peak amplitude (linear)
          - RMS amplitude (linear)

        All metrics calculated together to minimize iterations over samples.
        """
        self._processed = 0
        total: int = len(samples_list)

        def _worker(samples: np.ndarray, sr: int) -> Tuple[float, float, float, float]:
            frames, channels = samples.shape

            # R128 States
            st_i = R128State(channels, sr, MeasurementMode.MODE_I)           # LUFS
            st_lra = R128State(channels, sr, MeasurementMode.MODE_LRA)       # LRA
            st_tp = R128State(channels, sr, MeasurementMode.MODE_TRUE_PEAK)  # True peak

            sum_sq, max_abs_sq = 0.0, 0.0

            # Chunked loop
            for offset in range(0, frames, chunk_size):
                block = samples[offset:offset + chunk_size]
                interleaved = block.flatten()

                n = block.shape[0]
                st_i.add_frames(interleaved, n)
                st_lra.add_frames(interleaved, n)
                st_tp.add_frames(interleaved, n)

                # Accumulate peak and RMS in the same iteration
                sq_block = block ** 2
                sum_sq += np.sum(sq_block)
                block_max_sq = sq_block.max()
                if block_max_sq > max_abs_sq:
                    max_abs_sq = block_max_sq

            # Finalize metrics from R128 states
            lufs = get_loudness_global(st_i)
            lra = get_loudness_range(st_lra)
            tp_ch = [20 * math.log10(get_true_peak(st_tp, ch)) for ch in range(channels)]
            true_peak = max(tp_ch)

            # Compute final peak and RMS
            peak = np.sqrt(max_abs_sq)
            rms = np.sqrt(sum_sq / (frames * channels)) if frames > 0 else 0.0
            crest_db = 20.0 * np.log10(peak / rms)

            self._processed += 1

            self._logger.info(f"Processed {self._processed}/{total} ({self._processed/total*100:.2f}%) tracks.", separator=self._separator)

            return true_peak, lufs, lra, crest_db

        # Parallelize across tracks
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            results = exe.map(_worker, samples_list, sample_rates)

        tps, lufs, lras, crest_db = zip(*results)
        return {
            Header.True_Peak.value:       np.array(tps,  dtype=np.float32),
            Header.Integrated_LUFS.value:        np.array(lufs, dtype=np.float32),
            Header.Program_Dynamic_Range_LRA.value:  np.array(lras, dtype=np.float32),
            Header.Crest_Factor.value: np.array(crest_db, dtype=np.float32),
        }

    def calculate_batch_rest(
            self,
            infos: List[AudioStreamsInfoModel],
            paths: List[Path]
    ) -> Dict[str, np.ndarray]:
        # 1. Pull scalars into NumPy arrays
        durations    = np.array([i.duration          for i in infos], dtype=np.float32)
        bitrates     = np.array([i.bitrate           for i in infos], dtype=np.float32)
        sample_rates = np.array([i.sample_rate_kHz   for i in infos], dtype=np.float32)
        maxdps       = np.array([self._calculate_max_data_per_second(i) for i in infos],
                                dtype=np.float32)
        bit_depths = np.array(i.bit_depth for i in infos)

        # 2. Use the passed-in paths for size/duration
        actual_bits_per_sec = np.empty(len(paths), dtype=np.float32)
        for idx, (p, info) in enumerate(zip(paths, infos)):
            size_b    = os.path.getsize(p)
            size_bits = size_b * 8
            actual_bits_per_sec[idx] = size_bits / info.duration if info.duration > 0 else 0.0

        # 3. Compute efficiency
        effs = np.where(maxdps > 0,
                        actual_bits_per_sec / maxdps * 100.0,
                        0.0)

        return {
            Header.Duration.value:             durations,
            Header.Bitrate.value:              bitrates,
            Header.Sample_Rate.value:          sample_rates,
            Header.Max_Data_Per_Second.value:  maxdps / 1_000,
            Header.Actual_Data_Rate.value:     actual_bits_per_sec / 1_000,
            Header.Efficiency.value:           effs,
            Header.Bit_Depth.value: bit_depths,
        }

    def _calculate_max_data_per_second(self, stream_info: AudioStreamsInfoModel) -> float:
        """Calculates the data rate per second in bits."""
        if stream_info.bit_depth is None:
            return 0.0
        if stream_info.sample_rate_Hz == 0 or stream_info.bit_depth == 0 or stream_info.channels == 0:
            return 0.0

        cache_key: Tuple[float, int, int] = (stream_info.sample_rate_Hz, stream_info.bit_depth, stream_info.channels)

        cached_value = self._max_data_rate_cache_lookup.get(
            cache_key,
            None
        )

        if cached_value is not None:
            return cached_value

        data_rate_per_second = stream_info.sample_rate_Hz * stream_info.bit_depth * stream_info.channels
        self._max_data_rate_cache_lookup[cache_key] = data_rate_per_second

        self._logger.debug(f"Calculated data rate per second: {data_rate_per_second}", separator=self._separator)

        return data_rate_per_second

    def save_cache(self):
        """Persists the cache on disk. Call when finished with all calculations for optimization gains."""
        self._save_max_data_rate_cache_lookup()

    def _load_max_data_rate_cache_lookup(self) -> Dict[Tuple[float, int, int], float]:
        if not self._max_data_rate_cache_path.exists():
            self._logger.warning("Max data rate cache not found. Initializing empty cache.", separator=self._separator)
            return {}

        with open(self._max_data_rate_cache_path, "rb") as f:
            return pickle.load(f)

    def _save_max_data_rate_cache_lookup(self) -> None:
        with open(self._max_data_rate_cache_path, "wb") as f:
            # noinspection PyTypeChecker
            pickle.dump(self._max_data_rate_cache_lookup, f)
