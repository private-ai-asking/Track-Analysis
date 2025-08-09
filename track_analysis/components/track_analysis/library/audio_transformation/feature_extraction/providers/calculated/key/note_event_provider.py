from typing import Dict, Any, List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.preprocessing.note_extraction.note_event_builder import \
    NoteEventBuilder


class NoteEventProvider(AudioDataFeatureProvider):
    def __init__(self, logger: HoornLogger, hop_length: int = 512):
        super().__init__()
        self._note_event_builder: NoteEventBuilder = NoteEventBuilder(logger, hop_length=hop_length)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.TRACK_CLEANED_BINARY_MASK, AudioDataFeature.TRACK_MIDI_MAP, AudioDataFeature.SAMPLE_RATE_HZ]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return AudioDataFeature.TRACK_NOTE_EVENTS

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            cleaned_mask = data[AudioDataFeature.TRACK_CLEANED_BINARY_MASK]
            midi_map = data[AudioDataFeature.TRACK_MIDI_MAP]
            sample_rate = data[AudioDataFeature.SAMPLE_RATE_HZ]

            note_events = self._note_event_builder.build_note_events(
                cleaned_mask=cleaned_mask,
                midi_map=midi_map,
                sr=sample_rate,
            )

            return {
                AudioDataFeature.TRACK_NOTE_EVENTS: note_events,
            }
