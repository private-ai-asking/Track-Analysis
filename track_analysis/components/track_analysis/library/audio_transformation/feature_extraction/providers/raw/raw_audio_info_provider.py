from typing import List, Dict, Any

from pymediainfo import MediaInfo

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider


class RawAudioInfoProvider(AudioDataFeatureProvider):
    """
    A provider that loads audio and extracts raw information from a file path.
    """

    def __init__(self, logger):
        super().__init__()
        self._logger = logger

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH]

    @property
    def output_features(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.DURATION,
            AudioDataFeature.SAMPLE_RATE_HZ,
            AudioDataFeature.SAMPLE_RATE_KHZ,
            AudioDataFeature.BIT_DEPTH,
            AudioDataFeature.NUM_CHANNELS,
            AudioDataFeature.BIT_RATE,
        ]

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            audio_path = data[AudioDataFeature.AUDIO_PATH]

        with self._measure_waiting():
            media_info = MediaInfo.parse(str(audio_path))

        with self._measure_processing():
            audio_track = media_info.audio_tracks[0]

            duration_s = float(audio_track.duration) / 1000.0 if audio_track.duration else 0.0
            bitrate_bps = float(audio_track.bit_rate) if audio_track.bit_rate is not None else 0.0
            sample_rate = int(audio_track.sampling_rate) if audio_track.sampling_rate else 0
            bit_depth = float(audio_track.bit_depth) if audio_track.bit_depth else None
            channels = int(audio_track.channel_s) if audio_track.channel_s else 0

            return {
                AudioDataFeature.DURATION: duration_s,
                AudioDataFeature.BIT_RATE: bitrate_bps / 1000,
                AudioDataFeature.SAMPLE_RATE_HZ: sample_rate,
                AudioDataFeature.SAMPLE_RATE_KHZ: sample_rate / 1000,
                AudioDataFeature.BIT_DEPTH: bit_depth,
                AudioDataFeature.NUM_CHANNELS: channels,
            }
