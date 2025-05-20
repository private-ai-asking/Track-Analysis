import pickle
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from numpy import ndarray
from pyebur128.pyebur128 import get_loudness_global, R128State, MeasurementMode, get_true_peak, get_loudness_range

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import CACHE_DIRECTORY
from track_analysis.components.track_analysis.features.audio_file_handler import AudioStreamsInfoModel


class AudioCalculator:
    def __init__(self, logger: HoornLogger):
        self._separator = "AudioCalculator"
        self._logger = logger

        self._max_data_rate_cache_path: Path = CACHE_DIRECTORY / "max_data_rate_cache.pkl"
        self._max_data_rate_cache_lookup: Dict[Tuple[float, int, int], float] = self._load_max_data_rate_cache_lookup()

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def save_cache(self):
        """Persists the cache on disk. Call when finished with all calculations for optimization gains."""
        self._save_max_data_rate_cache_lookup()

    def calculate_program_dynamic_range_and_crest_factor(self,
                                                         samples: np.ndarray,
                                                         samplerate: int
                                                         ) -> Tuple[float, float]:
        float_frames: np.ndarray[np.float32] = np.ascontiguousarray(samples, dtype=np.float32).ravel()  # type: ignore

        crest_db: float = self._get_crest_factor(float_frames)
        program_dr_db: float = self._get_program_dynamic_range(float_frames, samples, samplerate)

        self._logger.debug(
            f"Program DR (LRA): {program_dr_db:.2f} dB, "
            f"Crest Factor: {crest_db:.2f} dB",
            separator=self._separator
        )
        return program_dr_db, crest_db

    def _get_crest_factor(self, samples: np.ndarray[np.float32]) -> float:
        peak, rms    = self._peak_and_rms(samples)
        ratio = peak / rms if rms != 0.0 else 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            crest_db = 20.0 * np.log10(ratio)

        return crest_db

    @staticmethod
    def _get_program_dynamic_range(float_frames: np.ndarray, samples: np.ndarray, samplerate: int) -> float:
        frames, nch = samples.shape

        state = R128State(channels=nch,
                          samplerate=samplerate,
                          mode=MeasurementMode.MODE_LRA)
        state.add_frames(float_frames, frames)
        program_dr_db = get_loudness_range(state)
        return program_dr_db

    def calculate_max_data_per_second(self, stream_info: AudioStreamsInfoModel) -> float:
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

    def calculate_lufs(self, sample_rate: float, samples: np.ndarray) -> float:
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

    def calculate_true_peak(
            self,
            sample_rate: float,
            samples: ndarray
    ) -> float:
        """
        Calculates the true-peak level in dBTP (≤ 0.0 dB).

        Args:
          sample_rate (float): Sample rate in Hz.
          samples     (ndarray): shape = (frames, channels) or (n,) for mono,
                                 dtype float64 with ±1.0 == full scale.

        Returns:
          float: True-peak in dBTP (–inf if completely silent).
        """
        # 1) Flatten to interleaved 1D
        flat = samples.flatten()

        # 2) Normalize full-scale to ±1.0 if any |sample| > 1.0
        max_val = np.max(np.abs(flat))
        if max_val > 1.0:
            flat = flat / max_val

        # 3) Determine channel count
        try:
            n_ch = samples.shape[1]
        except IndexError:
            n_ch = 1

        # 4) Build the true-peak state
        state = R128State(
            channels   = n_ch,
            samplerate = int(sample_rate),
            mode       = MeasurementMode.MODE_TRUE_PEAK
        )

        # 5) Feed all frames so it can oversample & interpolate
        n_frames = flat.size // n_ch
        state.add_frames(flat, n_frames)

        # 6) Get each channel’s raw true peak and pick the highest
        peaks = [ get_true_peak(state, ch) for ch in range(n_ch) ]
        tp_raw = max(peaks)

        # 7) Convert to dBTP, guarding against log(0)
        tp_db = 20 * np.log10(tp_raw) if tp_raw > 0 else -float('inf')

        self._logger.debug(f"True peak: {tp_db:.2f} dBTP", separator=self._separator)
        return tp_db

    @staticmethod
    def _peak_and_rms(arr: np.ndarray[np.float32]) -> Tuple[float, float]:
        peak = np.max(np.abs(arr))
        rms  = np.sqrt(np.mean(arr**2))
        return float(peak), float(rms)

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
