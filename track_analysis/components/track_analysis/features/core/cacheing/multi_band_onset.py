from pathlib import Path
import numpy as np
import librosa
from typing import Dict, Tuple, cast

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
    # 1) obtain (cached) magnitude spectrogram
    S = magnitude_extractor.extract(
        file_path=file_path,
        start_sample=start_sample,
        end_sample=end_sample,
        audio=audio
    )

    # 2) map bands to FFT-bin slices
    freqs = np.linspace(0, sample_rate / 2, S.shape[0])
    channels: list[slice] = []
    band_names: list[str] = []
    for name, (fmin, fmax) in bands:
        i0 = np.searchsorted(freqs, fmin)
        i1 = np.searchsorted(freqs, fmax if fmax is not None else sample_rate / 2)
        channels.append(slice(i0, i1))
        band_names.append(name)

    # 3) compute multi-band onset strengths
    onset_envs = librosa.onset.onset_strength_multi(
        S=S,
        sr=sample_rate,
        hop_length=hop_length,
        channels=channels
    )

    # 4) return dict: band name -> envelope
    return {band_names[idx]: onset_envs[idx] for idx in range(len(band_names))}


class OnsetStrengthMultiExtractor:
    """
    Extractor for per-band onset-strength envelopes, reusing the cached magnitude spectrogram and onset computation.
    """
    def __init__(
            self,
            logger: HoornLogger,
            magnitude_extractor: MagnitudeSpectrogramExtractor,
            bands: Dict[str, Tuple[float, float]]
    ):
        self._logger = logger
        self._magnitude_extractor = magnitude_extractor
        self._bands = bands
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

        # prepare bands tuple for cache key
        bands_tuple = cast(
            tuple[tuple[str, Tuple[float, float]], ...],
            tuple(self._bands.items())
        )

        # delegate to cached function
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
