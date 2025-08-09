from typing import List, Dict, Any

import numpy as np
from librosa import fft_frequencies

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.shared.caching.hdf5_memory import TimedCacheResult
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.timed_cache(identifier_arg="identifier")
def _compute_fft_frequencies(*, sample_rate: int, n_fft: int, identifier: str = "__FFT FREQUENCIES__") -> TimedCacheResult[np.ndarray]:
    """
    Cached FFT frequency computation.
    The cache key is (sample_rate, n_fft).
    """
    # noinspection PyTypeChecker
    return fft_frequencies(sr=sample_rate, n_fft=n_fft)

class FFTFrequenciesProvider(AudioDataFeatureProvider):
    """
    Provides FFT frequency bins, cached based on sample rate and n_fft.
    """
    def __init__(self, n_fft: int = 2048):
        super().__init__()
        self._n_fft = n_fft

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.SAMPLE_RATE_HZ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.FFT_FREQUENCIES

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            sample_rate = data[AudioDataFeature.SAMPLE_RATE_HZ]

        fft_freqs = _compute_fft_frequencies(
            sample_rate=sample_rate,
            n_fft=self._n_fft
        )
        self._add_timed_cache_times(fft_freqs)

        with self._measure_processing():
            return {AudioDataFeature.FFT_FREQUENCIES: fft_freqs.value}
