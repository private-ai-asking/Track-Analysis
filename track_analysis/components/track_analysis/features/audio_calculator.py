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
