from typing import Dict, Any, List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.calculator.magnitude_spectogram import \
    MagnitudeSpectrogramExtractor


class HarmonicSpectrogramProvider(AudioDataFeatureProvider):
    def __init__(self, logger: HoornLogger, hop_length=512, n_fft=2048):
        super().__init__()
        self._magnitude_extractor = MagnitudeSpectrogramExtractor(logger, hop_length=hop_length, n_fft=n_fft)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.HARMONIC_AUDIO]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return AudioDataFeature.HARMONIC_MAGNITUDE_SPECTROGRAM

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            audio_path = data[AudioDataFeature.AUDIO_PATH]
            harmonic_audio = data[AudioDataFeature.HARMONIC_AUDIO]

        magnitudes = self._magnitude_extractor.extract(
            file_path=audio_path,
            audio=harmonic_audio,
            start_sample=0,
            end_sample=len(harmonic_audio),
        )
        self._add_timed_cache_times(magnitudes)

        with self._measure_processing():
            return {
                AudioDataFeature.HARMONIC_MAGNITUDE_SPECTROGRAM: magnitudes.value,
            }
