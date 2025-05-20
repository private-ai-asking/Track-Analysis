import librosa
import numpy as np
import pyebur128
from numpy import ndarray
from pyebur128.pyebur128 import get_loudness_global, R128State, MeasurementMode

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

    def calculate_lufs(self, sr: float, samples: ndarray) -> float:
        # 1) Re-shape into frames × channels
        if samples.ndim == 1:
            n_channels = 1
            frames = samples.shape[0]
            interleaved = samples.astype(np.float64)
        else:
            n_channels, n_samples = samples.shape
            frames = n_samples
            interleaved = samples.T.reshape(-1).astype(np.float64)

        # 3) Fresh R128State for this buffer
        state = R128State(
            channels=n_channels,
            samplerate=int(sr),
            mode=MeasurementMode.MODE_I
        )

        # 4) Feed everything in one go
        state.add_frames(interleaved, frames)

        # 5) Get integrated loudness (calls ff_ebur128_loudness_global)
        lufs = get_loudness_global(state)

        if VERBOSE:
            self._logger.debug(f"Integrated LUFS: {lufs}", separator=self._separator)

        return lufs

    def calculate_true_peak(self, sample_rate: float, samples: ndarray, oversampling_rate: int = 4) -> float:
        # Oversample the audio
        samples_resampled = librosa.resample(samples, orig_sr=sample_rate, target_sr=sample_rate*oversampling_rate)

        # Calculate the peak amplitude
        peak_amplitude = np.max(np.abs(samples_resampled))

        # Convert to dBFS
        true_peak_dbfs = 20 * np.log10(peak_amplitude)

        self._logger.debug(f"Calculated True Peak (dBPT): {true_peak_dbfs}", separator=self._separator)
        return true_peak_dbfs
