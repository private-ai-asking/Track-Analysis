from typing import Dict, Any, List

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.features.audio_calculation.utils.cacheing.max_rate_cache import \
    MaxRateCache
from track_analysis.components.track_analysis.features.audio_calculation.utils.file_utils import FileUtils


class DataEfficiencyProvider(AudioDataFeatureProvider):
    def __init__(self, file_utils: FileUtils, rate_cache: MaxRateCache):
        self._file_utils = file_utils
        self._rate_cache = rate_cache

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.DURATION, AudioDataFeature.SAMPLE_RATE_HZ, AudioDataFeature.BIT_DEPTH, AudioDataFeature.NUM_CHANNELS]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return [AudioDataFeature.DATA_RATE, AudioDataFeature.MAX_DATA_RATE, AudioDataFeature.DATA_EFFICIENCY]

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        audio_path = data[AudioDataFeature.AUDIO_PATH]
        duration = data[AudioDataFeature.DURATION]
        sample_rate = data[AudioDataFeature.SAMPLE_RATE_HZ]
        bit_depth = data[AudioDataFeature.BIT_DEPTH]
        num_channels = data[AudioDataFeature.NUM_CHANNELS]

        actual_rate_bps = (self._file_utils.get_size_bytes(audio_path) * 8) / duration if duration > 0 else 0.0

        max_rate_bps = self._rate_cache.get(sample_rate, bit_depth, num_channels)
        efficiency = (actual_rate_bps / max_rate_bps * 100) if max_rate_bps > 0 else 0.0

        return {
            AudioDataFeature.DATA_RATE: actual_rate_bps,
            AudioDataFeature.MAX_DATA_RATE: max_rate_bps,
            AudioDataFeature.DATA_EFFICIENCY: efficiency,
        }
