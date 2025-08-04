from typing import Dict, Any, List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.features.core.cacheing.magnitude_spectogram import \
    MagnitudeSpectrogramExtractor


class HarmonicSpectrogramProvider(AudioDataFeatureProvider):
    def __init__(self, logger: HoornLogger, hop_length=512, n_fft=2048):
        self._magnitude_extractor = MagnitudeSpectrogramExtractor(logger, hop_length=hop_length, n_fft=n_fft)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.HARMONIC_AUDIO]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return AudioDataFeature.HARMONIC_MAGNITUDE_SPECTROGRAM

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        magnitudes = self._magnitude_extractor.extract(
            file_path=data[AudioDataFeature.AUDIO_PATH],
            audio=data[AudioDataFeature.HARMONIC_AUDIO],
            start_sample=0,
            end_sample=len(data[AudioDataFeature.HARMONIC_AUDIO]),
        )

        return {
            AudioDataFeature.HARMONIC_MAGNITUDE_SPECTROGRAM: magnitudes,
        }

