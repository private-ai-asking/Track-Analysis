import numpy as np
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

from numba import njit, prange

from track_analysis.components.track_analysis.features.audio_calculation.calculators.metric_calculator import \
    AudioMetricCalculator
from track_analysis.components.track_analysis.features.core.cacheing.harmonic import HarmonicExtractor
from track_analysis.components.track_analysis.features.core.cacheing.harmonicity import HarmonicityExtractor
from track_analysis.components.track_analysis.features.core.cacheing.magnitude_spectogram import \
    MagnitudeSpectrogramExtractor
from track_analysis.components.track_analysis.features.core.cacheing.mfcc import MfccExtractor
from track_analysis.components.track_analysis.features.core.cacheing.multi_band_onset import OnsetStrengthMultiExtractor
from track_analysis.components.track_analysis.features.core.cacheing.onset_envelope import OnsetStrengthExtractor
from track_analysis.components.track_analysis.features.core.cacheing.spectral_contrast import SpectralContrastExtractor
from track_analysis.components.track_analysis.features.core.cacheing.spectral_flatness import SpectralFlatnessExtractor
from track_analysis.components.track_analysis.features.core.cacheing.spectral_rolloff import SpectralRolloffExtractor
from track_analysis.components.track_analysis.features.core.cacheing.zero_crossing import ZeroCrossingRateExtractor


# --- Your Custom Helper Functions ---

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
    """
    n_bins, n_frames = S.shape
    for j in prange(n_frames):
        # Spectral Centroid
        num = 0.0
        den = 1e-8
        for i in range(n_bins):
            v = S[i, j]
            num += freqs[i] * v
            den += v
        cent_out[j] = num / den

        # Spectral Flux
        if j > 0:
            acc = 0.0
            for i in range(n_bins):
                d = S[i, j] - S[i, j - 1]
                if d > 0:
                    acc += d * d
            flux_out[j - 1] = acc ** 0.5

@lru_cache(maxsize=None)
def _get_freqs(sr: int, n_bins: int) -> np.ndarray:
    """
    Calculate and cache the frequency values for each bin of an FFT.
    """
    return np.linspace(0, sr / 2, n_bins)


# --- Dependency Container ---

@dataclass(frozen=True)
class FeatureExtractors:
    """A container for all feature extractor dependencies."""
    harmonic: HarmonicExtractor
    magnitude: MagnitudeSpectrogramExtractor
    onset_global: OnsetStrengthExtractor
    onset_multi: OnsetStrengthMultiExtractor
    zcr: ZeroCrossingRateExtractor
    flatness: SpectralFlatnessExtractor
    contrast: SpectralContrastExtractor
    rolloff: SpectralRolloffExtractor
    harmonicity: HarmonicityExtractor
    mfcc: MfccExtractor

# --- Refactored Calculator Class ---

class SpectralRhythmCalculator(AudioMetricCalculator):
    """
    Calculates spectral and rhythmic features from an audio signal using a
    structured, dependency-injected, and modular approach.
    """

    def __init__(self, extractors: FeatureExtractors, hop_length: int):
        self._extractors = extractors
        self._hop_length = hop_length

    def calculate(
            self,
            audio_path: Path,
            samples: np.ndarray,
            sr: int,
            **kwargs: Any
    ) -> Dict[str, float]:
        duration_sec = len(samples) / sr
        if duration_sec == 0:
            return {}

        harmonic, percussive = self._extractors.harmonic.extract_harmonic(
            file_path=audio_path,
            audio=samples,
            sample_rate=sr,
            tempo_bpm=kwargs.get("tempo")
        )

        base_args = {"file_path": audio_path, "sample_rate": sr, "hop_length": self._hop_length}

        harmonic_features = self._calculate_spectral_features(harmonic, sr, base_args)
        percussive_features = self._calculate_onset_features(percussive, duration_sec, base_args)
        full_signal_features = self._calculate_global_features(samples, base_args)

        return {
            "harmonicity":self._extractors.harmonicity.extract(**{
                "file_path": audio_path,
                "sample_rate": sr,
            }, start_sample=0, end_sample=len(samples), harmonic=harmonic, full_audio=samples),
            **harmonic_features,
            **percussive_features,
            **full_signal_features
        }

    def _calculate_spectral_features(
            self, harmonic: np.ndarray, sr: int, base_args: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculates features from the harmonic component using custom helpers."""
        common_args = {**base_args, "audio": harmonic, "start_sample": 0, "end_sample": len(harmonic)}

        s_h = self._extractors.magnitude.extract(**{
            "file_path": common_args["file_path"],
            "start_sample": common_args["start_sample"],
            "end_sample": common_args["end_sample"],
            "audio": common_args["audio"]
        })

        n_bins, n_frames = s_h.shape
        if n_frames < 2:  # Not enough frames to calculate flux
            return {}

        # 1. Get frequencies using your cached helper
        freqs = _get_freqs(sr, n_bins)

        # 2. Prepare output arrays for the in-place Numba function
        cent = np.empty(n_frames, dtype=np.float64)
        flux = np.empty(n_frames - 1, dtype=np.float64)

        # 3. Call your custom Numba function
        _centroid_and_flux(s_h, freqs, cent, flux)

        spectral_rolloff = self._extractors.rolloff.extract(**common_args, roll_percent=0.85)

        return {
            "spec_centroid_mean_hz": float(np.mean(cent)),
            "spec_centroid_max_hz": float(np.max(cent)),
            "spec_flux_mean": float(np.mean(flux)),
            "spec_flux_max": float(np.max(flux)),
            "spec_rolloff_mean": float(np.mean(spectral_rolloff)),
            "spec_rolloff_std": float(np.std(spectral_rolloff)),
        }

    def _calculate_onset_features(
            self, percussive: np.ndarray, duration_sec: float, base_args: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculates global and per-band onset features from the percussive component."""
        common_args = {**base_args, "audio": percussive, "start_sample": 0, "end_sample": len(percussive)}

        env_global = self._extractors.onset_global.extract(**common_args)
        peaks_global = self._extractors.onset_global.extract_peaks(**common_args)
        dynamic_tempo = self._extractors.onset_global.extract_dynamic_tempo(**common_args)

        results = {
            "onset_env_mean": float(env_global.mean()),
            "onset_rate": len(peaks_global) / duration_sec,
            "tempo_variation": np.std(dynamic_tempo),
        }

        onset_envs_multi = self._extractors.onset_multi.extract(**common_args)
        onset_peaks_multi = self._extractors.onset_multi.extract_peaks(**common_args)
        onset_means = {f"onset_env_mean_{name}": float(env.mean()) for name, env in onset_envs_multi.items()}
        onset_rates = {
            f"onset_rate_{name}": len(onset_peaks_multi.get(name, [])) / duration_sec
            for name in onset_envs_multi.keys()
        }

        return {**results, **onset_means, **onset_rates}

    def _calculate_global_features(
            self, samples: np.ndarray, base_args: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculates features from the full, original audio signal."""
        common_args = {**base_args, "audio": samples, "start_sample": 0, "end_sample": len(samples)}

        zcr = self._extractors.zcr.extract(**common_args)
        flatness = self._extractors.flatness.extract(**common_args)
        contrast = self._extractors.contrast.extract(**common_args)
        mffccs = self._extractors.mfcc.extract(**{
            "file_path": common_args["file_path"],
            "start_sample": common_args["start_sample"],
            "end_sample": common_args["end_sample"],
            "audio": common_args["audio"],
            "sample_rate": common_args["sample_rate"]
        }, n_mfcc=20)

        mffcc_means = np.mean(mffccs, axis=1)
        mfcc_stds = np.std(mffccs, axis=1)

        return {
            "zcr_mean": float(zcr.mean()),
            "spectral_flatness_mean": float(flatness.mean()),
            "spectral_contrast_mean": float(contrast.mean()),
            "mffcc_means": np.array(mffcc_means),
            "mfcc_stds": np.array(mfcc_stds),
        }
