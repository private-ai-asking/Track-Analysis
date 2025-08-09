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


@MEMORY.timed_cache(identifier_arg="file_path", ignore=["magnitude_spectrogram"])
def _compute_spectral_entropy(
        *,
        file_path: Path,
        magnitude_spectrogram: np.ndarray,
        unique_string: str,
) -> TimedCacheResult[float]:
    """
    Cached calculation for spectral entropy.
    """
    frame_entropies = entropy(magnitude_spectrogram + 1e-9, base=2, axis=0)

    # Return the mean entropy across all time frames
    return float(np.mean(frame_entropies)) # type: ignore


class SpectralEntropyProvider(AudioDataFeatureProvider):
    """
    Provides the spectral entropy, a measure of the "randomness" or "unpredictability"
    of the spectral content.
    """
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        # Also uses the harmonic spectrogram for textural analysis.
        return [
            AudioDataFeature.AUDIO_PATH,
            AudioDataFeature.HARMONIC_MAGNITUDE_SPECTROGRAM,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.SPECTRAL_ENTROPY

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            file_path = data[AudioDataFeature.AUDIO_PATH]
            harmonic_spec = data[AudioDataFeature.HARMONIC_MAGNITUDE_SPECTROGRAM]

        spec_entropy = _compute_spectral_entropy(
            file_path=file_path,
            magnitude_spectrogram=harmonic_spec,
            unique_string="harmonic_entropy"
        )
        self._add_timed_cache_times(spec_entropy)

        with self._measure_processing():
            spec_entropy = spec_entropy.value
            return {AudioDataFeature.SPECTRAL_ENTROPY: spec_entropy}
