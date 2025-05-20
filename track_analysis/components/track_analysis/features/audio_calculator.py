import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple, Dict, List, Union

import numpy as np
import samplerate
from numba import njit, prange
from numpy import ndarray
from pyebur128.pyebur128 import get_loudness_global, R128State, MeasurementMode, get_true_peak, get_loudness_range
from scipy.signal import resample_poly

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


def _tp_worker_resample(
        samples: np.ndarray,
        oversample_rate: int
) -> float:
    # noinspection PyUnresolvedReferences
    res = samplerate.resample(samples, oversample_rate, 'sinc_fastest')
    peak = np.max(np.abs(res))
    return 20 * np.log10(peak) if peak > 0 else -np.inf


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
            oversample_rate: int = 4,
            max_workers: Union[int, None] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute dBTP for a batch of tracks via fast C upsampling,
        parallelized across cores without using any lambdas.
        """
        n = len(samples_list)
        # build two parallel iterables:
        #  - one of your sample‐arrays
        #  - one of the same oversample_rate repeated
        rates = [oversample_rate] * n

        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            # map our top-level function over two iterables
            results = exe.map(_tp_worker_resample, samples_list, rates)

        # pack into your header dict
        return {
            Header.True_Peak.value: np.array(list(results), dtype=np.float32)
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
        lufs         = np.array([self._calculate_lufs(i.sample_rate_Hz, i.samples)
                                 for i in infos], dtype=np.float32)

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
            Header.Loudness.value:             lufs,
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

    def _calculate_lufs(self, sample_rate: float, samples: np.ndarray) -> float:
        # 1) Re-shape into interleaved frames×channels
        # Multichannel: shape == (frames, channels)
        frames, n_channels = samples.shape
        interleaved = samples.flatten()

        # 2) Build a fresh R128State
        state = R128State(
            channels   = n_channels,
            samplerate = int(sample_rate),
            mode       = MeasurementMode.MODE_I
        )

        # 3) Feed all frames at once
        state.add_frames(interleaved, frames)

        # 4) Compute integrated loudness
        lufs = get_loudness_global(state)

        self._logger.debug(f"Integrated LUFS: {lufs}", separator=self._separator)

        return lufs

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
