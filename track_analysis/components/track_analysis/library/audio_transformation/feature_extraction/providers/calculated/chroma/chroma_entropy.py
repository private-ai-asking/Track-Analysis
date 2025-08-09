from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from scipy.stats import entropy

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.shared.caching.hdf5_memory import TimedCacheResult
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.timed_cache(identifier_arg="file_path", ignore=["chromagram"])
def _compute_chroma_entropy(
        *,
        file_path: Path,
        chromagram: np.ndarray,
        unique_string: str,
) -> TimedCacheResult[float]:
    """
    Cached calculation for the mean entropy of a chromagram.
    """
    # Calculate entropy along the pitch axis (axis=0) for each time frame
    frame_entropies = entropy(chromagram + 1e-9, base=2, axis=0)

    # Return the mean entropy across all frames
    return float(np.mean(frame_entropies)) # type: ignore


class ChromaEntropyProvider(AudioDataFeatureProvider):
    """
    Provides the mean chroma entropy, a measure of pitch clarity.
    Low entropy indicates a clear, stable tonal center.
    High entropy indicates ambiguous, complex, or noisy harmony.
    """
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.AUDIO_PATH,
            AudioDataFeature.CHROMAGRAM,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.CHROMA_ENTROPY

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            file_path = data[AudioDataFeature.AUDIO_PATH]
            chroma = data[AudioDataFeature.CHROMAGRAM]

        chroma_entropy_result = _compute_chroma_entropy(
            file_path=file_path,
            chromagram=chroma,
            unique_string="harmonic_chroma_entropy"
        )

        self._add_timed_cache_times(chroma_entropy_result)

        with self._measure_processing():
            return {
                AudioDataFeature.CHROMA_ENTROPY: chroma_entropy_result.value
            }
