import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import soxr
import pyloudnorm as pyln
from numba import njit, prange
from pyebur128.pyebur128 import R128State, MeasurementMode, get_loudness_range

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import CACHE_DIRECTORY
from track_analysis.components.track_analysis.features.audio_file_handler import AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


@njit(parallel=True, fastmath=True, cache=True)
def _batch_peak_rms(
        data: np.ndarray,
        offsets: np.ndarray,
        counts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    n = counts.shape[0]
    peaks = np.empty(n, dtype=np.float32)
    rmss  = np.empty(n, dtype=np.float32)
    for i in prange(n):
        start = offsets[i]
        length = counts[i]
        max_sq = 0.0
        sum_sq = 0.0
        for j in range(start, start + length):
            x = data[j]
            sq = x * x
            sum_sq += sq
            if sq > max_sq:
                max_sq = sq
        peaks[i] = np.sqrt(max_sq)
        rmss[i]  = np.sqrt(sum_sq / length)
    return peaks, rmss


def _tp_worker_soxr(
        samples: np.ndarray,
        in_rate: int,
        oversample_rate: int,
        quality: str
) -> float:
    """
    Upsample `samples` from `in_rate` to `in_rate * oversample_rate`
    and compute dB True-Peak.
    """
    # ensure float32 for soxr
    data = samples.astype(np.float32)
    # resample in one shot; quality ∈ {"QQ","LQ","MQ","HQ","VHQ"}
    out = soxr.resample(data, in_rate, in_rate * oversample_rate, quality=quality)  # :contentReference[oaicite:0]{index=0}
    peak = np.max(np.abs(out))
    return 20.0 * np.log10(peak) if peak > 0 else -np.inf


class AudioCalculator:
    def __init__(self, logger: HoornLogger):
        self._separator = "AudioCalculator"
        self._logger = logger

        self._max_data_rate_cache_path: Path = CACHE_DIRECTORY / "max_data_rate_cache.pkl"
        self._max_data_rate_cache_lookup: Dict[Tuple[float, int, int], float] = self._load_max_data_rate_cache_lookup()

        self._logger.trace("Successfully initialized.", separator=self._separator)

    @staticmethod
    def calculate_batch_crest(
            samples_list: List[np.ndarray]   # each is shape=(frames,channels)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns two 1-D arrays of length N: (peak, rms) for each track.
        """
        # 1) flatten each track and record offsets
        flat_arrays = [arr.astype(np.float32).ravel() for arr in samples_list]
        counts      = np.array([a.size for a in flat_arrays], dtype=np.int64)
        offsets     = np.concatenate(([0], counts.cumsum()[:-1]))
        all_data    = np.concatenate(flat_arrays)

        # 2) call one Numba‐parallel routine
        peaks, rmss = _batch_peak_rms(all_data, offsets, counts)
        return peaks, rmss

    @staticmethod
    def calculate_batch_true_peak(
            samples_list: List[np.ndarray],
            sample_rates: List[float],
            oversample_rate: int = 4,
            quality: str = "MQ",
            max_workers: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-track dB True-Peak via libsoxr, in parallel.
        - `samples_list[i]` has sample rate `sample_rates[i]`.
        - `oversample_rate` of 4–8 is usually plenty; higher gives diminishing returns.
        - `quality` can be “LQ” or “MQ” for extra speed (see speed bench).
        """
        n = len(samples_list)
        # build parallel iterables
        rates = [oversample_rate] * n
        quals = [quality] * n

        with ThreadPoolExecutor(max_workers=max_workers) as exec:
            # map our top-level worker over four iterables
            results = exec.map(
                _tp_worker_soxr,
                samples_list,
                sample_rates,
                rates,
                quals
            )

        return {
            Header.True_Peak.value: np.fromiter(results, dtype=np.float32)
        }

    @staticmethod
    def calculate_batch_lufs(
            samples_list: List[np.ndarray],
            sample_rates: List[float],
            block_size: float        = 0.800,
            filter_class: str        = "Fenton/Lee 1",
            max_workers: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Batch‐measure integrated LUFS via pyloudnorm, in parallel.

        - Uses larger block_size (default 0.8 s) to cut filter calls in half.
        - Uses Fenton/Lee IIR filters for ~>2× speed with <0.1 dB error.
        - ThreadPoolExecutor so each call runs in C outside the GIL.
        """
        # 1) create one Meter per unique sample_rate
        meters: Dict[float, pyln.Meter] = {
            sr: pyln.Meter(sr, block_size=block_size, filter_class=filter_class)
            for sr in set(sample_rates)
        }

        # 2) worker just calls the prebuilt Meter
        def _measure(samples: np.ndarray, sr: int) -> float:
            # samples shape = (frames, channels)
            return meters[sr].integrated_loudness(samples)

        # 3) parallel map
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            results = exe.map(_measure, samples_list, sample_rates)

        return {
            Header.Integrated_LUFS.value: np.fromiter(results, dtype=np.float32)
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
        }

    def calculate_program_dr(self,
                             samples: np.ndarray,
                             samplerate: float
                             ) -> float:
        float_frames: np.ndarray[np.float32] = np.ascontiguousarray(samples, dtype=np.float32).ravel()  # type: ignore

        program_dr_db: float = self._get_program_dynamic_range(float_frames, samples, samplerate)

        self._logger.debug(
            f"Program DR (LRA): {program_dr_db:.2f} dB ",
            separator=self._separator
        )
        return program_dr_db

    @staticmethod
    def _get_program_dynamic_range(float_frames: np.ndarray, samples: np.ndarray, samplerate: float) -> float:
        frames, nch = samples.shape

        state = R128State(channels=nch,
                          samplerate=samplerate,
                          mode=MeasurementMode.MODE_LRA)
        state.add_frames(float_frames, frames)
        program_dr_db = get_loudness_range(state)
        return program_dr_db

    def _calculate_max_data_per_second(self, stream_info: AudioStreamsInfoModel) -> float:
        """Calculates the data rate per second in bits."""
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
