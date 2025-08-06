from typing import List, Dict, Any

import numpy as np

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.helpers.spectral_helper import \
    spectral_get_freqs, spectral_centroid_and_flux


class SpectralCentroidAndFluxProvider(AudioDataFeatureProvider):
    """
    Intermediate calculator that computes the raw spectral centroid and flux arrays.
    The final mean/max calculators will depend on this.
    """
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.HARMONIC_MAGNITUDE_SPECTROGRAM, AudioDataFeature.SAMPLE_RATE_HZ]

    @property
    def output_features(self) -> List[AudioDataFeature]:
        # This calculator outputs two intermediate features
        return [AudioDataFeature.SPECTRAL_CENTROID_ARRAY, AudioDataFeature.SPECTRAL_FLUX_ARRAY]

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, np.ndarray]:
        s_h = data[AudioDataFeature.HARMONIC_MAGNITUDE_SPECTROGRAM]
        sr = data[AudioDataFeature.SAMPLE_RATE_HZ]
        n_bins, n_frames = s_h.shape

        if n_frames < 2:
            return {
                AudioDataFeature.SPECTRAL_CENTROID_ARRAY: np.array([]),
                AudioDataFeature.SPECTRAL_FLUX_ARRAY: np.array([])
            }

        freqs = spectral_get_freqs(sr, n_bins)
        cent_out = np.empty(n_frames, dtype=np.float64)
        flux_out = np.empty(n_frames - 1, dtype=np.float64)

        spectral_centroid_and_flux(s_h, freqs, cent_out, flux_out)

        return {
            AudioDataFeature.SPECTRAL_CENTROID_ARRAY: cent_out,
            AudioDataFeature.SPECTRAL_FLUX_ARRAY: flux_out
        }
