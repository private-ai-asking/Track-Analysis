from typing import List, Dict, Any

import numpy as np

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_calculator import \
    AudioDataFeatureCalculator
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.features.core.cacheing.harmonic import HarmonicExtractor


class HPSSExtractor(AudioDataFeatureCalculator):
    """Calculates and returns the harmonic and percussive components of an audio signal."""
    def __init__(self, logger, hop_length=512, n_fft=2048):
        self._harmonic_extractor = HarmonicExtractor(logger, hop_length, n_fft)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.AUDIO_SAMPLES, AudioDataFeature.AUDIO_SAMPLE_RATE, AudioDataFeature.BPM]

    # This calculator produces MULTIPLE outputs. The orchestrator can be adapted to handle this,
    # or you can have two separate classes. For simplicity, we'll assume the orchestrator can handle it.
    @property
    def output_features(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.HARMONIC_AUDIO, AudioDataFeature.PERCUSSIVE_AUDIO]

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, np.ndarray]:
        harmonic, percussive = self._harmonic_extractor.extract_harmonic(
            file_path=data[AudioDataFeature.AUDIO_PATH],
            audio=data[AudioDataFeature.AUDIO_SAMPLES],
            sample_rate=data[AudioDataFeature.AUDIO_SAMPLE_RATE],
            tempo_bpm=data[AudioDataFeature.BPM]
        )
        return {
            AudioDataFeature.HARMONIC_AUDIO: harmonic,
            AudioDataFeature.PERCUSSIVE_AUDIO: percussive
        }
