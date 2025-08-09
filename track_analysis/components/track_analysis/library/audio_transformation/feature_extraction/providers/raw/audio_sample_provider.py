from typing import Dict, Any, List

import soundfile as sf

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider


class AudioSampleProvider(AudioDataFeatureProvider):
    def __init__(self, logger: HoornLogger):
        super().__init__()
        self._logger = logger
        self._separator = self.__class__.__name__

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.SAMPLE_RATE_HZ]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_SAMPLES]

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            audio_path = data[AudioDataFeature.AUDIO_PATH]
            original_sr = data[AudioDataFeature.SAMPLE_RATE_HZ]

        with self._measure_waiting():
            samples, sr = sf.read(audio_path, dtype='float32')

        with self._measure_processing():
            if sr != original_sr:
                self._logger.warning(
                    f"Sample-rate mismatch for {audio_path}: "
                    f"media_info={original_sr} Hz vs soundfile={sr} Hz",
                    separator=self._separator
                )

            if samples.ndim > 1:
                samples = samples.mean(axis=1)

            return {
                AudioDataFeature.AUDIO_SAMPLES: samples
            }
