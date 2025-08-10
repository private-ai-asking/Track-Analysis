import dataclasses
import time
from pathlib import Path

import numpy as np

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.calculator.mfcc import \
    MfccExtractor
from track_analysis.components.track_analysis.shared.caching.hdf5_memory import TimedCacheResult


@dataclasses.dataclass(frozen=True)
class MFCCResult:
    means:        np.ndarray
    stds:         np.ndarray
    delta_means:  np.ndarray
    delta_stds:   np.ndarray
    delta2_means: np.ndarray
    delta2_stds:  np.ndarray

class MFCCHelper:
    def __init__(self, mfcc_extractor: MfccExtractor, number_of_mfccs: int):
        self._mfcc_extractor: MfccExtractor = mfcc_extractor
        self._number_of_mfccs = number_of_mfccs

    def get_mffcs(self,
                  audio_path: Path,
                  audio: np.ndarray,
                  sample_rate: int) -> TimedCacheResult[MFCCResult]:
        mffccs = self._mfcc_extractor.extract_mfccs(file_path=audio_path, start_sample=0, end_sample=len(audio), sample_rate=sample_rate, n_mfcc=self._number_of_mfccs, audio=audio)
        mffccs_value = mffccs.value

        deltas = self._mfcc_extractor.extract_deltas(file_path=audio_path, start_sample=0, end_sample=len(audio), sample_rate=sample_rate, order=1, axis=-1, mffccs=mffccs_value)
        deltas_value = deltas.value

        delta2s = self._mfcc_extractor.extract_deltas(file_path=audio_path, start_sample=0, end_sample=len(audio), sample_rate=sample_rate, order=2, axis=-1, mffccs=mffccs_value)
        delta2s_value = delta2s.value

        processing_start = time.perf_counter()
        mffcc_means = np.mean(mffccs_value, axis=1)
        mfcc_stds = np.std(mffccs_value, axis=1)

        delta_means  = np.mean(deltas_value,  axis=1)
        delta_stds   = np.std( deltas_value,  axis=1)
        delta2_means = np.mean(delta2s_value, axis=1)
        delta2_stds  = np.std( delta2s_value, axis=1)

        result = MFCCResult(
            means=mffcc_means,
            stds=mfcc_stds,
            delta_means=delta_means,
            delta_stds=delta_stds,
            delta2_means=delta2_means,
            delta2_stds=delta2_stds,
        )
        processing_duration = time.perf_counter() - processing_start

        return TimedCacheResult(
            value=result,
            time_processing=mffccs.time_processing+deltas.time_processing+delta2s.time_processing+processing_duration,
            time_waiting=mffccs.time_waiting+deltas.time_waiting+delta2s.time_waiting,
            retrieved_from_cache=False
        )
