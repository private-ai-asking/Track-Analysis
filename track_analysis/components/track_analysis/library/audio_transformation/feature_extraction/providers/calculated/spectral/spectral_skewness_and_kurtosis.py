from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
from scipy.stats import skew, kurtosis

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.shared.caching.hdf5_memory import TimedCacheResult
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.timed_cache(identifier_arg="file_path", ignore=["magnitude_spectrogram"])
def _compute_spectral_moments(
        *,
        file_path: Path,
        magnitude_spectrogram: np.ndarray,
        unique_string: str,
) -> TimedCacheResult[Tuple[float, float]]:
    """
    Cached calculation for spectral skewness and kurtosis.
    These are calculated across the time axis for each frequency bin, then averaged.
    """
    # Calculate skewness and kurtosis along the time axis (axis=1)
    # We add a small epsilon to avoid issues with silent frames
    spec_skew = skew(magnitude_spectrogram + 1e-9, axis=1)
    spec_kurt = kurtosis(magnitude_spectrogram + 1e-9, axis=1)

    # Return the mean of the values across all frequency bins
    return float(np.mean(spec_skew)), float(np.mean(spec_kurt)) # type: ignore


class SpectralMomentsProvider(AudioDataFeatureProvider):
    """
    Provides statistical moments (skewness, kurtosis) of the harmonic spectrum.
    These features describe the shape of the spectral distribution, adding textural information.
    """
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.AUDIO_PATH,
            AudioDataFeature.HARMONIC_MAGNITUDE_SPECTROGRAM,
        ]

    @property
    def output_features(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.SPECTRAL_SKEWNESS,
            AudioDataFeature.SPECTRAL_KURTOSIS,
        ]

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            file_path = data[AudioDataFeature.AUDIO_PATH]
            harmonic_spec = data[AudioDataFeature.HARMONIC_MAGNITUDE_SPECTROGRAM]

        result = _compute_spectral_moments(
            file_path=file_path,
            magnitude_spectrogram=harmonic_spec,
            unique_string="harmonic_moments"
        )
        self._add_timed_cache_times(result)

        with self._measure_processing():
            skewness, kurt = result.value
            return {
                AudioDataFeature.SPECTRAL_SKEWNESS: skewness,
                AudioDataFeature.SPECTRAL_KURTOSIS: kurt,
            }
