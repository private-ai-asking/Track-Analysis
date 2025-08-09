from typing import List, Dict, Any

import numpy as np

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.separation.calculator.harmonic import \
    HarmonicExtractor


class HPSSExtractor(AudioDataFeatureProvider):
    """Calculates and returns the harmonic and percussive components of an audio signal."""
    def __init__(self, logger, hop_length=512, n_fft=2048):
        super().__init__()
        self._harmonic_extractor = HarmonicExtractor(logger, hop_length, n_fft)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.AUDIO_SAMPLES, AudioDataFeature.SAMPLE_RATE_HZ, AudioDataFeature.BPM]

    @property
    def output_features(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.HARMONIC_AUDIO, AudioDataFeature.PERCUSSIVE_AUDIO]

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, np.ndarray]:
        result = self._harmonic_extractor.extract_harmonic(
            file_path=data[AudioDataFeature.AUDIO_PATH],
            audio=data[AudioDataFeature.AUDIO_SAMPLES],
            sample_rate=data[AudioDataFeature.SAMPLE_RATE_HZ],
            tempo_bpm=data[AudioDataFeature.BPM]
        )

        self._add_timed_cache_times(result)

        with self._measure_processing():
            harmonic, percussive = result.value
            return {
                AudioDataFeature.HARMONIC_AUDIO: harmonic,
                AudioDataFeature.PERCUSSIVE_AUDIO: percussive
            }
