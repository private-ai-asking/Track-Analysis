from pathlib import Path
from typing import Tuple

import numpy as np

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.calculator.mfcc import \
    MfccExtractor


class MFCCHelper:
    def __init__(self, mfcc_extractor: MfccExtractor):
        self._mfcc_extractor: MfccExtractor = mfcc_extractor

    def get_mffcs(self,
                  audio_path: Path,
                  audio: np.ndarray,
                  sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        mffccs = self._mfcc_extractor.extract(file_path=audio_path, start_sample=0, end_sample=len(audio), sample_rate=sample_rate, n_mfcc=20, audio=audio)

        mffcc_means = np.mean(mffccs, axis=1)
        mfcc_stds = np.std(mffccs, axis=1)

        return mffcc_means, mfcc_stds
