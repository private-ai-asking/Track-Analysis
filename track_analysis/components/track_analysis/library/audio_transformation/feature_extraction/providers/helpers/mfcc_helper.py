import dataclasses
from pathlib import Path
from typing import Tuple

import numpy as np

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.calculator.mfcc import \
    MfccExtractor

@dataclasses.dataclass(frozen=True)
class MFCCResult:
    means:        np.ndarray
    stds:         np.ndarray
    delta_means:  np.ndarray
    delta_stds:   np.ndarray
    delta2_means: np.ndarray
    delta2_stds:  np.ndarray

class MFCCHelper:
    def __init__(self, mfcc_extractor: MfccExtractor):
        self._mfcc_extractor: MfccExtractor = mfcc_extractor

    def get_mffcs(self,
                  audio_path: Path,
                  audio: np.ndarray,
                  sample_rate: int) -> MFCCResult:
        mffccs = self._mfcc_extractor.extract_mfccs(file_path=audio_path, start_sample=0, end_sample=len(audio), sample_rate=sample_rate, n_mfcc=20, audio=audio)
        deltas = self._mfcc_extractor.extract_deltas(file_path=audio_path, start_sample=0, end_sample=len(audio), sample_rate=sample_rate, order=1, axis=-1, mffccs=mffccs)
        delta2s = self._mfcc_extractor.extract_deltas(file_path=audio_path, start_sample=0, end_sample=len(audio), sample_rate=sample_rate, order=2, axis=-1, mffccs=mffccs)

        mffcc_means = np.mean(mffccs, axis=1)
        mfcc_stds = np.std(mffccs, axis=1)

        delta_means  = np.mean(deltas,  axis=1)
        delta_stds   = np.std( deltas,  axis=1)
        delta2_means = np.mean(delta2s, axis=1)
        delta2_stds  = np.std( delta2s, axis=1)

        return MFCCResult(
            means=mffcc_means,
            stds=mfcc_stds,
            delta_means=delta_means,
            delta_stds=delta_stds,
            delta2_means=delta2_means,
            delta2_stds=delta2_stds,
        )
