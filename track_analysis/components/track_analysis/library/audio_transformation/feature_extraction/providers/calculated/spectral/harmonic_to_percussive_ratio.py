from pathlib import Path
from typing import List, Any, Dict

import numpy as np

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.loudness.calculator.rms import \
    compute_linear_rms_cached


class HarmonicToPercussiveRatioProvider(AudioDataFeatureProvider):
    """
    Calculates the ratio of the total energy of the harmonic component to the
    total energy of the percussive component by reusing a cached linear RMS function.

    A high value indicates a predominantly harmonic/tonal track (e.g., ambient).
    A low value indicates a track with significant percussive content.
    """
    def __init__(self, window_ms: float = 50.0, hop_ms: float = 10.0):
        super().__init__()
        self._window_ms = window_ms
        self._hop_ms = hop_ms

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.AUDIO_PATH,
            AudioDataFeature.HARMONIC_AUDIO,
            AudioDataFeature.PERCUSSIVE_AUDIO,
            AudioDataFeature.SAMPLE_RATE_HZ,
            AudioDataFeature.AUDIO_SAMPLES,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.HPR

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            file_path: Path = data[AudioDataFeature.AUDIO_PATH]
            harmonic_audio: np.ndarray = data[AudioDataFeature.HARMONIC_AUDIO]
            percussive_audio: np.ndarray = data[AudioDataFeature.PERCUSSIVE_AUDIO]
            sample_rate: int = data[AudioDataFeature.SAMPLE_RATE_HZ]
            start_sample = 0
            end_sample = len(data[AudioDataFeature.AUDIO_SAMPLES])

        # 1. Calculate the RMS values for each component using the cached function.
        #    A unique `method_string` ensures separate cache entries for each component.
        rms_harmonic = compute_linear_rms_cached(
            file_path=file_path,
            audio=harmonic_audio,
            sample_rate=sample_rate,
            method_string="hpr_harmonic",
            start_sample=start_sample,
            end_sample=end_sample,
            window_ms=self._window_ms,
            hop_ms=self._hop_ms
        )
        self._add_timed_cache_times(rms_harmonic)

        rms_percussive = compute_linear_rms_cached(
            file_path=file_path,
            audio=percussive_audio,
            sample_rate=sample_rate,
            method_string="hpr_percussive",
            start_sample=start_sample,
            end_sample=end_sample,
            window_ms=self._window_ms,
            hop_ms=self._hop_ms
        )
        self._add_timed_cache_times(rms_percussive)

        with self._measure_processing():
            # 2. Calculate the mean square energy from the RMS values.
            harmonic_energy = np.mean(rms_harmonic.value**2)
            percussive_energy = np.mean(rms_percussive.value**2)

            # 3. Handle the edge case where percussive energy is zero.
            if percussive_energy < 1e-9:
                hpr = 1e9
            else:
                hpr = harmonic_energy / percussive_energy

            return {AudioDataFeature.HPR: hpr}
