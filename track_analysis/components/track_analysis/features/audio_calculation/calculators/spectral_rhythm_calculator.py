from functools import lru_cache
from pathlib import Path
from typing import Dict

import numpy as np
from numba import njit, prange

from track_analysis.components.track_analysis.features.audio_calculation.calculators.metric_calculator import \
    AudioMetricCalculator
from track_analysis.components.track_analysis.features.core.cacheing.harmonic import HarmonicExtractor
from track_analysis.components.track_analysis.features.core.cacheing.magnitude_spectogram import \
    MagnitudeSpectrogramExtractor
from track_analysis.components.track_analysis.features.core.cacheing.multi_band_onset import OnsetStrengthMultiExtractor
from track_analysis.components.track_analysis.features.core.cacheing.onset_envelope import OnsetStrengthExtractor


# noinspection t
@njit(parallel=True)
def _centroid_and_flux(
        S: np.ndarray,
        freqs: np.ndarray,
        cent_out: np.ndarray,
        flux_out: np.ndarray
) -> None:
    """
    Compute spectral centroid and flux in-place, in parallel.
    """
    n_bins, n_frames = S.shape
    for j in prange(n_frames):
        # centroid
        num = 0.0
        den = 1e-8
        for i in range(n_bins):
            v = S[i, j]
            num += freqs[i] * v
            den += v
        cent_out[j] = num / den

        # flux
        acc = 0.0
        if j > 0:
            for i in range(n_bins):
                d = S[i, j] - S[i, j - 1]
                if d > 0:
                    acc += d * d
        flux_out[j - 1] = acc ** 0.5


@lru_cache(maxsize=None)
def _get_freqs(sr: int, n_bins: int) -> np.ndarray:
    """
    Cache the frequency vector for a given sample rate and bin count.
    """
    return np.linspace(0, sr / 2, n_bins)


class SpectralRhythmCalculator(AudioMetricCalculator):
    """
    Calculates spectral centroid, spectral flux, and per-band onset-rate metrics.
    """
    def __init__(
            self,
            harmonic_extractor: HarmonicExtractor,
            magnitude_extractor: MagnitudeSpectrogramExtractor,
            onset_extractor: OnsetStrengthExtractor,
            onset_extractor_multi: OnsetStrengthMultiExtractor,
            hop_length: int,
    ):
        self._harmonic_extractor = harmonic_extractor
        self._magnitude_extractor = magnitude_extractor
        self._onset_extractor = onset_extractor
        self._onset_extractor_multi = onset_extractor_multi
        self._hop_length = hop_length

    def calculate(
            self,
            audio_path: Path,
            samples: np.ndarray,
            sr: int,
            **kwargs
    ) -> Dict[str, float]:
        # 1) split
        tempo = kwargs.get("tempo")
        harmonic, percussive = self._harmonic_extractor.extract_harmonic(
            file_path=audio_path, audio=samples, sample_rate=sr, tempo_bpm=tempo
        )

        # 2) centroid / flux (unchanged) …
        S_h = self._magnitude_extractor.extract(
            file_path=audio_path, audio=harmonic, start_sample=0, end_sample=harmonic.shape[0]
        )
        n_bins, n_frames = S_h.shape
        freqs = _get_freqs(sr, n_bins)
        cent  = np.empty(n_frames,       dtype=np.float64)
        flux  = np.empty(n_frames - 1,   dtype=np.float64)
        _centroid_and_flux(S_h, freqs, cent, flux)

        # 3) global onset (full-band)
        env_global = self._onset_extractor.extract(
            file_path=audio_path,
            start_sample=0,
            end_sample=percussive.shape[0],
            sample_rate=sr,
            hop_length=self._hop_length,
            audio=percussive
        )
        duration_sec = samples.shape[0] / sr
        mean_global = float(env_global.mean())
        rate_global = float((env_global > np.percentile(env_global, 75)).sum() / duration_sec)

        # 4) per-band onsets
        onset_envs = self._onset_extractor_multi.extract(
            file_path=audio_path,
            start_sample=0,
            end_sample=percussive.shape[0],
            sample_rate=sr,
            hop_length=self._hop_length,
            audio=percussive
        )
        onset_means = {}
        onset_rates = {}
        for name, env in onset_envs.items():
            onset_means[f"onset_env_mean_{name}"] = float(env.mean())
            onset_rates[f"onset_rate_{name}"]      = float((env > np.percentile(env, 75)).sum() / duration_sec)

        # 5) return everything
        return {
            "spec_centroid_mean_hz":   float(cent.mean()),
            "spec_centroid_max_hz":    float(cent.max()),
            "spec_flux_mean":          float(flux.mean()),
            "spec_flux_max":           float(flux.max()),

            "onset_env_mean":          mean_global,
            "onset_rate":              rate_global,

            # your new per-band metrics
            **onset_means,
            **onset_rates,
        }
