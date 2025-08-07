from typing import Dict, Any, List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.calculator.pitch_class_normalizer import \
    PitchClassesNormalizer


class NormalizedPitchClassesMaskProvider(AudioDataFeatureProvider):
    def __init__(self, logger: HoornLogger):
        self._pitch_class_normalizer = PitchClassesNormalizer(logger)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.TRACK_PITCH_CLASSES]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return AudioDataFeature.TRACK_NORMALIZED_PITCH_CLASSES_MASK

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        audio_path = data[AudioDataFeature.AUDIO_PATH]
        pitch_classes = data[AudioDataFeature.TRACK_PITCH_CLASSES]

        normalized_pitch_classes = self._pitch_class_normalizer.normalize_pitch_classes(
            file_path=audio_path,
            pitch_classes=pitch_classes,
        )

        return {
            AudioDataFeature.TRACK_NORMALIZED_PITCH_CLASSES_MASK: normalized_pitch_classes,
        }
