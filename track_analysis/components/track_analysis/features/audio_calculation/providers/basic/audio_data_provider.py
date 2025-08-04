from typing import Dict, Any, List

import librosa

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_provider import \
    AudioDataFeatureProvider


class AudioDataProvider(AudioDataFeatureProvider):
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = self.__class__.__name__

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.SAMPLE_RATE_HZ]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_SAMPLES]

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        audio_path = data[AudioDataFeature.AUDIO_PATH]
        original_sr = data[AudioDataFeature.SAMPLE_RATE_HZ]

        samples, sr = librosa.load(audio_path, sr=None)

        if sr != original_sr:
            self._logger.warning(
                f"Sample-rate mismatch for {audio_path}: "
                f"media_info={original_sr} Hz vs librosa={sr} Hz",
                separator=self._separator
            )

        return {
            AudioDataFeature.AUDIO_SAMPLES: samples
        }
