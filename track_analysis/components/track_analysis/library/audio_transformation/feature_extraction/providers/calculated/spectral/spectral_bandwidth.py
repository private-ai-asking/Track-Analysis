from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from librosa.feature import spectral_bandwidth

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.cache(identifier_arg="file_path", ignore=["magnitude_spectrogram", "centroid_array"])
def _compute_spectral_bandwidth(
        *,
        file_path: Path,
        magnitude_spectrogram: np.ndarray,
        centroid_array: np.ndarray,
        unique_string: str,
) -> np.ndarray:
    """
    Cached calculation for the spectral bandwidth.
    This reuses a pre-computed spectrogram and spectral centroid for efficiency.
    """
    if centroid_array.ndim == 1:
        centroid_array = np.expand_dims(centroid_array, axis=0)

    return spectral_bandwidth(
        S=magnitude_spectrogram,
        centroid=centroid_array,
    )[0]


# --- Provider Class ---

class SpectralBandwidthProvider(AudioDataFeatureProvider):
    """
    Provides the mean and standard deviation of the spectral bandwidth.

    Spectral bandwidth measures how "spread out" the spectral energy is around its
    centroid. A high value indicates a rich or noisy sound, while a low value
    indicates a "pure" or focused sound.
    """
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.AUDIO_PATH,
            AudioDataFeature.HARMONIC_MAGNITUDE_SPECTROGRAM,
            AudioDataFeature.SPECTRAL_CENTROID_ARRAY,
        ]

    @property
    def output_features(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.SPECTRAL_BANDWIDTH_MEAN,
            AudioDataFeature.SPECTRAL_BANDWIDTH_STD,
        ]

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        file_path = data[AudioDataFeature.AUDIO_PATH]
        harmonic_spec = data[AudioDataFeature.HARMONIC_MAGNITUDE_SPECTROGRAM]
        centroid_array = data[AudioDataFeature.SPECTRAL_CENTROID_ARRAY]

        # Get the frame-by-frame bandwidth values from the cached function
        bandwidth_array = _compute_spectral_bandwidth(
            file_path=file_path,
            magnitude_spectrogram=harmonic_spec,
            centroid_array=centroid_array,
            unique_string="harmonic_bandwidth"
        )

        # Calculate the final mean and standard deviation features
        mean_bandwidth = np.mean(bandwidth_array)
        std_bandwidth = np.std(bandwidth_array)

        return {
            AudioDataFeature.SPECTRAL_BANDWIDTH_MEAN: mean_bandwidth,
            AudioDataFeature.SPECTRAL_BANDWIDTH_STD: std_bandwidth,
        }
