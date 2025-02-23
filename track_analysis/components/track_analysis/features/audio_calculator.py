from typing import List, Tuple

import librosa
import numpy as np
import pyloudnorm
from numpy import ndarray

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import VERBOSE
from track_analysis.components.track_analysis.features.audio_file_handler import AudioStreamsInfoModel


class AudioCalculator:
    def __init__(self, logger: HoornLogger):
        self._separator = "AudioCalculator"

        self._logger = logger
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def calculate_dynamic_range_and_crest_factor(self, samples: ndarray) -> (float, float):
        """Calculates the peak-to-RMS dynamic range of an audio signal.

         Args:
             samples (ndarray): An ndarray of the audio signals.

         Returns:
             float: The dynamic range in dB, or None if an error occurs.
         """
        peak_amplitude = np.max(np.abs(samples))
        rms_amplitude = np.sqrt(np.mean(samples ** 2))

        if rms_amplitude == 0:
            return float('inf') if peak_amplitude > 0 else -float('inf')
        dynamic_range = 20 * np.log10(peak_amplitude / rms_amplitude)
        crest_factor = peak_amplitude / rms_amplitude

        self._logger.debug(f"Calculated dr & cf: {dynamic_range} & {crest_factor}", separator=self._separator)

        return dynamic_range, crest_factor

    def calculate_max_data_per_second(self, stream_info: AudioStreamsInfoModel) -> float:
        """Calculates the data rate per second in bits."""
        if stream_info.sample_rate_kHz == 0 or stream_info.bit_depth == 0 or stream_info.channels == 0:
            return 0.0

        data_rate_per_second = (stream_info.sample_rate_kHz * 1000) * stream_info.bit_depth * stream_info.channels

        self._logger.debug(f"Calculated data rate per second: {data_rate_per_second}", separator=self._separator)

        return data_rate_per_second

    def calculate_lufs(self, sample_rate: float, samples: ndarray) -> float:
        if VERBOSE:
            self._logger.debug(f"Sample rate: {sample_rate}", separator=self._separator)

        meter = pyloudnorm.Meter(sample_rate)
        loudness = meter.integrated_loudness(samples)

        self._logger.debug(f"Calculated Loudness: {loudness}", separator=self._separator)
        return loudness

    def calculate_true_peak(self, sample_rate: float, samples: ndarray, oversampling_rate: int = 4) -> float:
        # Oversample the audio
        samples_resampled = librosa.resample(samples, orig_sr=sample_rate, target_sr=sample_rate*oversampling_rate)

        # Calculate the peak amplitude
        peak_amplitude = np.max(np.abs(samples_resampled))

        # Convert to dBFS
        true_peak_dbfs = 20 * np.log10(peak_amplitude)

        self._logger.debug(f"Calculated True Peak (dBPT): {true_peak_dbfs}", separator=self._separator)
        return true_peak_dbfs

    def _calculate_lufs_with_sliding_window(self, window_length_seconds: float, sample_rate: float, samples: ndarray) -> List[Tuple[float, float]]:
        """
        Calculates LUFS values with a sliding window approach and associates each LUFS value with a timestamp.

        Args:
            window_length_seconds (float): The length of the sliding window in seconds.
            sample_rate (float): The sample rate of the audio.
            samples (ndarray): The audio samples as a NumPy array.

        Returns:
            List[Tuple[float, float]]: A list of tuples, where each tuple contains the timestamp (in seconds) of the start of the window
                                     and the corresponding LUFS value.
        """
        meter = pyloudnorm.Meter(sample_rate)
        window_size_samples = int(window_length_seconds * sample_rate)

        # Ensure window size is not larger than the audio length
        if window_size_samples > len(samples):
            self._logger.warning("Window size is larger than the audio length. Reducing window size.", separator=self._separator)
            window_size_samples = len(samples)

        lufs_values_with_timestamps = []

        # Iterate through the audio data using a sliding window
        for i in range(0, len(samples) - window_size_samples + 1):
            window = samples[i:i + window_size_samples]
            timestamp = i / sample_rate  # Calculate the timestamp in seconds
            try:
                momentary_lufs = meter.integrated_loudness(window)
                lufs_values_with_timestamps.append((timestamp, momentary_lufs))
            except Exception as e:
                self._logger.warning(f"Error calculating loudness for window at {i}: {e}", separator=self._separator)
                lufs_values_with_timestamps.append((timestamp, 0.0))  # Use 0.0 as a default LUFS value in case of error

        return lufs_values_with_timestamps

    def calculate_momentary_short_term_lufs(self, sample_rate: float, samples: ndarray) -> (List[Tuple[float, float]], List[Tuple[float, float]]):
        """Calculates momentary and short-term LUFS of an audio file."""
        self._logger.trace(f"Calculating time series lufs with sample rate: {sample_rate}", separator=self._separator)

        try:
            # Calculate momentary LUFS
            momentary_lufs = self._calculate_lufs_with_sliding_window(0.4, sample_rate, samples)

            # Calculate short-term LUFS
            short_term_lufs = self._calculate_lufs_with_sliding_window(3, sample_rate, samples)

            return momentary_lufs, short_term_lufs

        except Exception as e:
            self._logger.warning(f"Error calculating LUFS: {e}", separator=self._separator)
            return None, None
