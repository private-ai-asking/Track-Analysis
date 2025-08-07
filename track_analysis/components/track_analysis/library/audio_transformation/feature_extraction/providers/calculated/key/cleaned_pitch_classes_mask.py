from typing import Dict, Any, List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.calculator.pitch_class_cleaner import \
    NormalizedPitchClassesCleaner


class TrackCleanedPitchClassesMaskProvider(AudioDataFeatureProvider):
    def __init__(self, logger: HoornLogger, n_fft: int = 2048, hop_length: int = 512):
        self._cleaner = NormalizedPitchClassesCleaner(logger, n_fft=n_fft, hop_length=hop_length)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.AUDIO_SAMPLES, AudioDataFeature.SAMPLE_RATE_HZ, AudioDataFeature.TRACK_NORMALIZED_PITCH_CLASSES_MASK]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return [AudioDataFeature.TRACK_CLEANED_BINARY_MASK, AudioDataFeature.TRACK_CLEANED_CHROMA_MASK]

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        audio_path = data[AudioDataFeature.AUDIO_PATH]
        audio_samples = data[AudioDataFeature.AUDIO_SAMPLES]
        sample_rate = data[AudioDataFeature.SAMPLE_RATE_HZ]
        normalized_pitch_classes_mask = data[AudioDataFeature.TRACK_NORMALIZED_PITCH_CLASSES_MASK]

        cleaned_binary, cleaned_chroma = self._cleaner.clean(
            file_path=audio_path,
            normalized=normalized_pitch_classes_mask,
            raw_audio_samples=audio_samples,
            sample_rate=sample_rate
        )

        return {
            AudioDataFeature.TRACK_CLEANED_BINARY_MASK: cleaned_binary,
            AudioDataFeature.TRACK_CLEANED_CHROMA_MASK: cleaned_chroma
        }
