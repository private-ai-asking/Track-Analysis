from pathlib import Path
from typing import Tuple, Dict, Any, List

import librosa
import numpy as np

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.cache(identifier_arg="file_path", ignore=["spectral_data"])
def _compute_spectral_peaks(
        *,
        file_path: Path,
        sample_rate: int,
        min_frq: int,
        max_frq: int,
        hop_length: int,
        n_fft: int,
        spectral_data: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cached spectral peak extraction via librosa.piptrack:
    - Cache key: (file_path, start_sample, end_sample, sample_rate, min_frq, max_frq, hop_length, n_fft)
    - `spectral_data` is ignored in the cache key but used if provided.
    """
    spectrogram = spectral_data

    # Extract pitches and magnitudes
    pitches, magnitudes = librosa.piptrack(
        S=spectrogram,
        sr=sample_rate,
        fmin=min_frq,
        fmax=max_frq,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    return pitches, magnitudes

class SpectralPeakProvider(AudioDataFeatureProvider):
    def __init__(self, min_frequency_hz: int, max_frequency_hz: int, hop_length: int, n_fft: int):
        self._min_frequency_hz = min_frequency_hz
        self._max_frequency_hz = max_frequency_hz
        self._hop_length = hop_length
        self._n_fft = n_fft

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.SAMPLE_RATE_HZ, AudioDataFeature.HARMONIC_MAGNITUDE_SPECTROGRAM]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return [AudioDataFeature.SPECTRAL_PITCH_ARRAY, AudioDataFeature.SPECTRAL_MAGNITUDES_ARRAY]

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        audio_path = data[AudioDataFeature.AUDIO_PATH]
        sample_rate = data[AudioDataFeature.SAMPLE_RATE_HZ]
        spectrogram = data[AudioDataFeature.HARMONIC_MAGNITUDE_SPECTROGRAM]

        pitches, magnitudes = _compute_spectral_peaks(
            file_path=audio_path,
            sample_rate=sample_rate,
            min_frq=self._min_frequency_hz,
            max_frq=self._max_frequency_hz,
            hop_length=self._hop_length,
            n_fft=self._n_fft,
            spectral_data=spectrogram,
        )

        return {
            AudioDataFeature.SPECTRAL_PITCH_ARRAY: pitches,
            AudioDataFeature.SPECTRAL_MAGNITUDES_ARRAY: magnitudes,
        }
