from typing import Dict, Any, List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.calculator.midi_to_pitch import \
    MidiToPitchClassesConverter


class TrackPitchClassesProvider(AudioDataFeatureProvider):
    def __init__(self, logger: HoornLogger):
        self._pitch_class_builder = MidiToPitchClassesConverter(logger)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.TRACK_MIDI_MAP]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return [AudioDataFeature.TRACK_PITCH_CLASSES]

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        audio_path = data[AudioDataFeature.AUDIO_PATH]
        midi_map = data[AudioDataFeature.TRACK_MIDI_MAP]

        pitch_classes = self._pitch_class_builder.convert(
            audio_path=audio_path,
            midi=midi_map,
        )

        return {
            AudioDataFeature.TRACK_PITCH_CLASSES: pitch_classes,
        }
