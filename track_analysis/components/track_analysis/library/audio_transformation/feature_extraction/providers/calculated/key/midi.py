from typing import Dict, Any, List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.calculator.frequency_to_midi import \
    FrequencyToMidi


class TrackMidiMapProvider(AudioDataFeatureProvider):
    def __init__(self, logger: HoornLogger):
        self._frequency_to_midi_map_converter = FrequencyToMidi(logger)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.SAMPLE_RATE_HZ, AudioDataFeature.SPECTRAL_PITCH_ARRAY, AudioDataFeature.SPECTRAL_MAGNITUDES_ARRAY]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return AudioDataFeature.TRACK_MIDI_MAP

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        audio_path = data[AudioDataFeature.AUDIO_PATH]
        sample_rate = data[AudioDataFeature.SAMPLE_RATE_HZ]
        frequencies = data[AudioDataFeature.SPECTRAL_PITCH_ARRAY]
        spectral_magnitudes = data[AudioDataFeature.SPECTRAL_MAGNITUDES_ARRAY]

        midi = self._frequency_to_midi_map_converter.convert(
            file_path=audio_path,
            sample_rate=sample_rate,
            frequencies=frequencies,
            magnitudes=spectral_magnitudes,
        )

        return {
            AudioDataFeature.TRACK_MIDI_MAP: midi,
        }
