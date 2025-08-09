import time
from pathlib import Path

import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.loudness.calculator.rms import \
    compute_linear_rms_cached
from track_analysis.components.track_analysis.shared.caching.hdf5_memory import TimedCacheResult
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.timed_cache(identifier_arg="file_path", ignore=["rms_harmonic_vals", "rms_total_vals"])
def compute_harmonicity(
        *,
        file_path:     Path,
        start_sample:  int,
        end_sample:    int,
        sample_rate:   int,
        rms_harmonic_vals: np.ndarray,
        rms_total_vals: np.ndarray,
) -> TimedCacheResult[float]:
    """
    Calculates the harmonicity ratio for the given range.
    Caches on (file_path, start_sample, end_sample, sample_rate, window_ms, hop_ms) only.
    """
    mean_rms_total = np.mean(rms_total_vals)
    if mean_rms_total > 0:
        harmonicity = np.mean(rms_harmonic_vals) / mean_rms_total
    else:
        harmonicity = 0.0

    return harmonicity

class HarmonicityExtractor:
    def __init__(self, logger: HoornLogger):
        self._logger    = logger
        self._separator = self.__class__.__name__

    def extract(
            self,
            file_path:     Path,
            start_sample:  int,
            end_sample:    int,
            sample_rate:   int,
            harmonic: np.ndarray,
            full_audio: np.ndarray,
            window_ms:     float = 50.0,
            hop_ms:        float = 10.0,
    ) -> TimedCacheResult[float]:
        """
        Returns the harmonicity ratio for the given range.
        """
        processing_start = time.perf_counter()
        y_harmonic = harmonic[start_sample:end_sample]
        y_audio = full_audio[start_sample:end_sample]
        processing_duration = time.perf_counter() - processing_start

        rms_harmonic_vals = compute_linear_rms_cached(file_path=file_path, audio=y_harmonic, sample_rate=sample_rate, window_ms=window_ms, hop_ms=hop_ms, start_sample=0, end_sample=len(y_harmonic), method_string="harmonic")
        rms_total_vals    = compute_linear_rms_cached(file_path=file_path, audio=y_audio, sample_rate=sample_rate, window_ms=window_ms, hop_ms=hop_ms, start_sample=0, end_sample=len(y_audio), method_string="full-audio")

        self._logger.debug(
            f"Extracting harmonicity for {file_path.name}[{start_sample}:{end_sample}]",
            separator=self._separator,
        )
        results = compute_harmonicity(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            rms_harmonic_vals=rms_harmonic_vals.value,
            rms_total_vals=rms_total_vals.value,
        )

        return TimedCacheResult(
            value = results.value,
            time_processing=processing_duration+rms_harmonic_vals.time_processing+rms_total_vals.time_processing+results.time_processing,
            time_waiting=rms_harmonic_vals.time_waiting+rms_total_vals.time_waiting+results.time_waiting,
            retrieved_from_cache=results.retrieved_from_cache,
        )
