from pathlib import Path
import numpy as np
import librosa
from typing import Dict, Tuple, cast

# Assuming these are placeholder imports for a larger project structure.
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.core.cacheing.shared import MEMORY
from track_analysis.components.track_analysis.features.core.cacheing.magnitude_spectogram import MagnitudeSpectrogramExtractor


@MEMORY.cache(ignore=["audio", "magnitude_extractor"])
def compute_onset_strengths_multi(
        *,
        file_path: Path,
        start_sample: int,
        end_sample: int,
        sample_rate: int,
        hop_length: int,
        bands: tuple[tuple[str, Tuple[float, float]], ...],
        magnitude_extractor: MagnitudeSpectrogramExtractor,
        audio: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """
    Returns per-band onset-strength envelopes for the given range.
    Caches on (file_path, start_sample, end_sample, sample_rate, hop_length, bands) only.
    Ignores the magnitude_extractor and audio in cache key.

    Uses the extractor to get a cached spectrogram, then computes onset strengths.
    """
    # 1) Obtain (cached) magnitude spectrogram.
    S = magnitude_extractor.extract(
        file_path=file_path,
        start_sample=start_sample,
        end_sample=end_sample,
        audio=audio
    )

    # 2) Map frequency bands to FFT-bin slices.
    freqs = np.linspace(0, sample_rate / 2, S.shape[0])
    channels: list[slice] = []
    band_names: list[str] = []
    for name, (fmin, fmax) in bands:
        i0 = np.searchsorted(freqs, fmin)
        i1 = np.searchsorted(freqs, fmax if fmax is not None else sample_rate / 2)
        channels.append(slice(i0, i1))
        band_names.append(name)

    # 3) Compute multi-band onset strengths using librosa.
    onset_envs = librosa.onset.onset_strength_multi(
        S=S,
        sr=sample_rate,
        hop_length=hop_length,
        channels=channels
    )

    # 4) Return a dictionary mapping band names to their respective envelopes.
    return {band_names[idx]: onset_envs[idx] for idx in range(len(band_names))}


@MEMORY.cache(ignore=["audio", "magnitude_extractor"])
def compute_onset_peaks_multi(
        *,
        file_path: Path,
        start_sample: int,
        end_sample: int,
        sample_rate: int,
        hop_length: int,
        bands: tuple[tuple[str, Tuple[float, float]], ...],
        magnitude_extractor: MagnitudeSpectrogramExtractor,
        audio: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """
    NEW: Computes and caches onset peaks for each band.
    This function first gets the cached onset envelopes and then finds the peaks in each one.
    """
    # 1) Get the (cached) onset strength envelopes.
    onset_envelopes = compute_onset_strengths_multi(
        file_path=file_path,
        start_sample=start_sample,
        end_sample=end_sample,
        sample_rate=sample_rate,
        hop_length=hop_length,
        bands=bands,
        magnitude_extractor=magnitude_extractor,
        audio=audio
    )

    # 2) Detect peaks in each band's envelope.
    onset_peaks = {}
    for name, onset_env in onset_envelopes.items():
        peaks = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=hop_length,
            units='frames'
        )
        onset_peaks[name] = peaks

    return onset_peaks


class OnsetStrengthMultiExtractor:
    """
    Extractor for per-band onset-strength envelopes and their peaks,
    reusing the cached magnitude spectrogram and onset computations.
    """
    def __init__(
            self,
            logger: HoornLogger,
            magnitude_extractor: MagnitudeSpectrogramExtractor
    ):
        self._logger = logger
        self._magnitude_extractor = magnitude_extractor
        self._bands = {
            # Kick / sub-kick fundamental energy
            "kick":   (20.0,   150.0),
            # Snare fundamentals & body
            "snare":  (150.0,  2500.0),
            # Toms / low-mid percussion (claps, wood blocks, congas)
            "low_mid":(2500.0, 5000.0),
            # Hi-hats, rides, cymbals
            "hihat":  (5000.0, None),
        }
        self._separator = self.__class__.__name__

    def extract(
            self,
            file_path: Path,
            start_sample: int,
            end_sample: int,
            sample_rate: int,
            hop_length: int,
            audio: np.ndarray | None = None,
    ) -> Dict[str, np.ndarray]:
        """
        Returns a dict of onset-strength envelopes, one per configured band.
        """
        self._logger.debug(
            f"Extracting multi-band onsets for {file_path.name}[{start_sample}:{end_sample}]",
            separator=self._separator
        )
        bands_tuple = cast(
            tuple[tuple[str, Tuple[float, float]], ...],
            tuple(self._bands.items())
        )
        return compute_onset_strengths_multi(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            hop_length=hop_length,
            bands=bands_tuple,
            magnitude_extractor=self._magnitude_extractor,
            audio=audio
        )

    def extract_peaks(
            self,
            file_path: Path,
            start_sample: int,
            end_sample: int,
            sample_rate: int,
            hop_length: int,
            audio: np.ndarray | None = None,
    ) -> Dict[str, np.ndarray]:
        self._logger.debug(
            f"Extracting multi-band onset peaks for {file_path.name}[{start_sample}:{end_sample}]",
            separator=self._separator
        )
        bands_tuple = cast(
            tuple[tuple[str, Tuple[float, float]], ...],
            tuple(self._bands.items())
        )
        return compute_onset_peaks_multi(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            hop_length=hop_length,
            bands=bands_tuple,
            magnitude_extractor=self._magnitude_extractor,
            audio=audio
        )
