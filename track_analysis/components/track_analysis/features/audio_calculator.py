from typing import Tuple

import numpy as np
from numpy import ndarray
from pyebur128.pyebur128 import get_loudness_global, R128State, MeasurementMode, get_true_peak, get_loudness_range

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.audio_file_handler import AudioStreamsInfoModel


class AudioCalculator:
    def __init__(self, logger: HoornLogger):
        self._separator = "AudioCalculator"

        self._logger = logger
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def calculate_program_dynamic_range_and_crest_factor(
            self,
            samples: np.ndarray,
            samplerate: int
    ) -> Tuple[float, float]:
        # --- 1) Program DR via EBU R128 LRA:
        frames, nch = samples.shape
        # pyebur128 wants float arrays
        float_frames = samples.flatten()
        state = R128State(channels=nch,
                          samplerate=samplerate,
                          mode=MeasurementMode.MODE_LRA)
        state.add_frames(float_frames, frames)
        program_dr_db = get_loudness_range(state)  # in LU = dB

        # --- 2) Crest Factor in dB (peak vs. RMS):
        # Replace any NaN or Inf just in case:
        flat = np.nan_to_num(float_frames, nan=0.0, posinf=0.0, neginf=0.0)
        peak = np.max(np.abs(flat))
        rms  = np.sqrt(np.mean(flat**2))
        if rms == 0:
            crest_db = float('inf') if peak > 0 else -float('inf')
        else:
            crest_db = 20 * np.log10(peak / rms)

        self._logger.debug(
            f"Program DR (LRA): {program_dr_db:.2f} dB, "
            f"Crest Factor: {crest_db:.2f} dB",
            separator=self._separator
        )
        return program_dr_db, crest_db

    def calculate_max_data_per_second(self, stream_info: AudioStreamsInfoModel) -> float:
        """Calculates the data rate per second in bits."""
        if stream_info.sample_rate_kHz == 0 or stream_info.bit_depth == 0 or stream_info.channels == 0:
            return 0.0

        data_rate_per_second = (stream_info.sample_rate_kHz * 1000) * stream_info.bit_depth * stream_info.channels

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
