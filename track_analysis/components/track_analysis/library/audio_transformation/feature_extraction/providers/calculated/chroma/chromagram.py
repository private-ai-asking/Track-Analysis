from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from librosa.feature import chroma_stft

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.cache(identifier_arg="file_path", ignore=["audio"])
def _compute_chromagram(
        *,
        file_path: Path,
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int,
        unique_string: str,
) -> np.ndarray:
    """
    Cached calculation for a chromagram from the harmonic component.
    """
    return chroma_stft(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length
    )

class ChromagramProvider(AudioDataFeatureProvider):
    """
    Provides a chromagram from the harmonic component of the audio.
    This is a prerequisite for calculating chroma-based features like entropy.
    """
    def __init__(self, hop_length: int = 512):
        self._hop_length = hop_length

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.AUDIO_PATH,
            AudioDataFeature.HARMONIC_AUDIO,
            AudioDataFeature.SAMPLE_RATE_HZ,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.CHROMAGRAM

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        chroma = _compute_chromagram(
            file_path=data[AudioDataFeature.AUDIO_PATH],
            audio=data[AudioDataFeature.HARMONIC_AUDIO],
            sample_rate=data[AudioDataFeature.SAMPLE_RATE_HZ],
            hop_length=self._hop_length,
            unique_string="harmonic_chroma"
        )
        return {AudioDataFeature.CHROMAGRAM: chroma}
