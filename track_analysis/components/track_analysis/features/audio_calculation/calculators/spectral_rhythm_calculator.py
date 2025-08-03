from functools import lru_cache
from pathlib import Path
from typing import Dict

import numpy as np
from numba import njit, prange

from track_analysis.components.track_analysis.features.audio_calculation.calculators.metric_calculator import AudioMetricCalculator
from track_analysis.components.track_analysis.features.core.cacheing.harmonic import HarmonicExtractor
from track_analysis.components.track_analysis.features.core.cacheing.magnitude_spectogram import MagnitudeSpectrogramExtractor
from track_analysis.components.track_analysis.features.core.cacheing.multi_band_onset import OnsetStrengthMultiExtractor
from track_analysis.components.track_analysis.features.core.cacheing.onset_envelope import OnsetStrengthExtractor
from track_analysis.components.track_analysis.features.core.cacheing.zero_crossing import ZeroCrossingRateExtractor
from track_analysis.components.track_analysis.features.core.cacheing.spectral_flatness import SpectralFlatnessExtractor
from track_analysis.components.track_analysis.features.core.cacheing.spectral_contrast import SpectralContrastExtractor


# noinspection t
@njit(parallel=True)
def _centroid_and_flux(
        S: np.ndarray,
        freqs: np.ndarray,
        cent_out: np.ndarray,
        flux_out: np.ndarray
) -> None:
    """
    Compute spectral centroid and flux in-place, in parallel using Numba.
    This function is efficient and its logic is sound.

    Args:
        S: Magnitude spectrogram (n_bins x n_frames).
        freqs: Frequencies for each bin.
        cent_out: Output array for centroid values (n_frames).
        flux_out: Output array for flux values (n_frames - 1).
    """
    n_bins, n_frames = S.shape
    for j in prange(n_frames):
        # --- Spectral Centroid ---
        # Weighted average of frequencies present in the signal.
        num = 0.0
        den = 1e-8  # Add a small epsilon to avoid division by zero.
        for i in range(n_bins):
            v = S[i, j]
            num += freqs[i] * v
            den += v
        cent_out[j] = num / den

        # --- Spectral Flux ---
        # Measures the change in the spectrum between adjacent frames.
        if j > 0:
            acc = 0.0
            for i in range(n_bins):
                # Calculate the difference and sum the squared positive changes.
                d = S[i, j] - S[i, j - 1]
                if d > 0:
                    acc += d * d
            flux_out[j - 1] = acc ** 0.5


@lru_cache(maxsize=None)
def _get_freqs(sr: int, n_bins: int) -> np.ndarray:
    """
    Calculate and cache the frequency values for each bin of an FFT.
    Caching prevents recalculation for tracks with the same sample rate and n_fft.
    """
    return np.linspace(0, sr / 2, n_bins)


class SpectralRhythmCalculator(AudioMetricCalculator):
    """
    Calculates spectral and rhythmic features from an audio signal.

    This revised version corrects a logical error in multi-band onset rate
    calculation and removes a redundant operation.

    Features calculated:
     - Spectral Centroid & Flux (from the harmonic component)
     - Onset Envelope & Rate (global and per-band, from the percussive component)
     - Zero-Crossing Rate
     - Spectral Flatness
     - Spectral Contrast
    """

    def __init__(
            self,
            harmonic_extractor: HarmonicExtractor,
            magnitude_extractor: MagnitudeSpectrogramExtractor,
            onset_extractor: OnsetStrengthExtractor,
            onset_extractor_multi: OnsetStrengthMultiExtractor,
            zcr_extractor: ZeroCrossingRateExtractor,
            flatness_extractor: SpectralFlatnessExtractor,
            contrast_extractor: SpectralContrastExtractor,
            hop_length: int,
    ):
        """Initializes the calculator with feature extractor dependencies."""
        self._harmonic_extractor = harmonic_extractor
        self._magnitude_extractor = magnitude_extractor
        self._onset_extractor = onset_extractor
        self._onset_extractor_multi = onset_extractor_multi
        self._zcr_extractor = zcr_extractor
        self._flatness_extractor = flatness_extractor
        self._contrast_extractor = contrast_extractor
        self._hop_length = hop_length

    def calculate(
            self,
            audio_path: Path,
            samples: np.ndarray,
            sr: int,
            **kwargs
    ) -> Dict[str, float]:
        """
        Processes the audio to compute all spectral and rhythmic metrics.

        Args:
            audio_path: Path to the audio file (used for caching).
            samples: The raw audio waveform.
            sr: The sample rate of the audio.
            **kwargs: Additional arguments, such as 'tempo'.

        Returns:
            A dictionary of computed audio metrics.
        """
        # --- 1. Initial Setup and Harmonic-Percussive Separation ---
        duration_sec = samples.shape[0] / sr
        if duration_sec == 0:
            return {} # Avoid division by zero for empty audio.

        harmonic, percussive = self._harmonic_extractor.extract_harmonic(
            file_path=audio_path,
            audio=samples,
            sample_rate=sr,
            tempo_bpm=kwargs.get("tempo")
        )

        # --- 2. Spectral Centroid & Flux (on harmonic component) ---
        S_h = self._magnitude_extractor.extract(
            file_path=audio_path,
            audio=harmonic,
            start_sample=0,
            end_sample=harmonic.shape[0]
        )
        n_bins, n_frames = S_h.shape
        freqs = _get_freqs(sr, n_bins)
        cent = np.empty(n_frames, dtype=np.float64)
        flux = np.empty(n_frames - 1, dtype=np.float64)
        _centroid_and_flux(S_h, freqs, cent, flux)

        # --- 3. Global Onset Envelope & Rate (on percussive component) ---
        common_args = {
            "file_path": audio_path,
            "start_sample": 0,
            "end_sample": percussive.shape[0],
            "sample_rate": sr,
            "hop_length": self._hop_length,
            "audio": percussive
        }
        env_global = self._onset_extractor.extract(**common_args)
        peaks_global = self._onset_extractor.extract_peaks(**common_args)
        rate_global = len(peaks_global) / duration_sec

        # --- 4. Multi-band Onsets (on percussive component) ---
        onset_envs = self._onset_extractor_multi.extract(**common_args)
        onset_peaks_multi = self._onset_extractor_multi.extract_peaks(**common_args)

        onset_means = {}
        onset_rates = {}
        for name, env in onset_envs.items():
            onset_means[f"onset_env_mean_{name}"] = float(env.mean())
            # Correctly calculate the rate for each band using its specific peaks.
            band_peaks = onset_peaks_multi.get(name, [])
            onset_rates[f"onset_rate_{name}"] = len(band_peaks) / duration_sec

        # --- 5. Other Features (on the full, original audio) ---
        full_audio_args = {
            "file_path": audio_path,
            "start_sample": 0,
            "end_sample": samples.shape[0],
            "sample_rate": sr,
            "hop_length": self._hop_length,
            "audio": samples
        }
        zcr = self._zcr_extractor.extract(**full_audio_args)
        flatness = self._flatness_extractor.extract(**full_audio_args)
        contrast = self._contrast_extractor.extract(**full_audio_args)

        # --- 6. Aggregate and Return Results ---
        results = {
            "spec_centroid_mean_hz": float(cent.mean()),
            "spec_centroid_max_hz": float(cent.max()),
            "spec_flux_mean": float(flux.mean()),
            "spec_flux_max": float(flux.max()),
            "onset_env_mean": float(env_global.mean()),
            "onset_rate": rate_global,
            "zcr_mean": float(zcr.mean()),
            "spectral_flatness_mean": float(flatness.mean()),
            "spectral_contrast_mean": float(contrast.mean()),
        }
        results.update(onset_means)
        results.update(onset_rates)

        return results
