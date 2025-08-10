import pprint
from pathlib import Path
from typing import Tuple, Any, List

import librosa
import numpy as np
from librosa.feature import rms
from scipy.signal import medfilt
from skimage.morphology import binary_opening, binary_closing

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.shared.caching.hdf5_memory import TimedCacheResult
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.timed_cache(identifier_arg="file_path", ignore=["data"])
def _compute_rms_envelope(
        file_path: Path,
        data: np.ndarray,
        n_fft: int,
        hop_length: int,
) -> TimedCacheResult[np.ndarray]:
    result = rms(
        y=data,
        frame_length=n_fft,
        hop_length=hop_length,
        center=True
    )[0]

    envelope_db = librosa.amplitude_to_db(result, ref=np.max)
    return envelope_db # type: ignore

@MEMORY.timed_cache(identifier_arg="file_path", ignore=["a", "b"])
def _get_perc_diff(file_path: Path, a: np.ndarray, b: np.ndarray) -> TimedCacheResult[float]:
    diffs = ~np.isclose(a, b, rtol=1e-5, atol=1e-8)
    n_changed = np.count_nonzero(diffs)
    perc_changed = 100 * (n_changed / a.size)
    return perc_changed # type: ignore

@MEMORY.timed_cache(identifier_arg="file_path", ignore=["envelope_db", "normalized"])
def _first_clean_pass(file_path: Path, envelope_db: np.ndarray, normalized: np.ndarray) -> TimedCacheResult[np.ndarray]:
    mask = (envelope_db >= -60)  # boolean (T=above noise-floor)
    cleaned_pass1 = normalized * mask[np.newaxis, :]
    return cleaned_pass1 # type: ignore

@MEMORY.timed_cache(identifier_arg="file_path", ignore=["cleaned_first_pass"])
def _second_clean_pass(file_path: Path, cleaned_first_pass: np.ndarray) -> TimedCacheResult[np.ndarray]:
    return medfilt(cleaned_first_pass, kernel_size=(1, 5)) # type: ignore

@MEMORY.timed_cache(identifier_arg="file_path", ignore=["cleaned_second_pass"])
def _third_clean_pass(file_path: Path, cleaned_second_pass: np.ndarray, threshold_db: int = -20) -> TimedCacheResult[np.ndarray]:
    cleaned_second_pass_db = librosa.amplitude_to_db(cleaned_second_pass, ref=np.max)
    return (cleaned_second_pass_db > threshold_db).astype(np.uint8) # type: ignore

@MEMORY.timed_cache(identifier_arg="file_path", ignore=["cleaned_third_pass"])
def _fourth_clean_pass(file_path: Path, cleaned_third_pass: np.ndarray) -> TimedCacheResult[np.ndarray]:
    # make sure we’re working with booleans
    mask = cleaned_third_pass.astype(bool)

    # 1×3 structuring element
    footprint = np.ones((1, 5), dtype=bool)

    # opening to remove tiny spikes, then closing to fill tiny holes
    opened = binary_opening(mask, footprint=footprint)
    closed = binary_closing(opened, footprint=footprint)

    # return as the same dtype you expect downstream
    return closed.astype(np.uint8)

@MEMORY.timed_cache(identifier_arg="file_path", ignore=["cleaned_fourth_pass"])
def _fifth_clean_pass(file_path: Path, cleaned_fourth_pass: np.ndarray, sr: int, hop_length: int) -> TimedCacheResult[np.ndarray]:
    min_len = int(0.075 * sr / hop_length)
    clean = np.zeros_like(cleaned_fourth_pass)
    for i in range(cleaned_fourth_pass.shape[0]):
        on_off = np.diff(np.concatenate([[0], cleaned_fourth_pass[i], [0]]))  # type: ignore
        starts = np.where(on_off==1)[0]
        ends   = np.where(on_off==-1)[0]
        for s,e in zip(starts, ends):
            if (e-s) >= min_len:
                clean[i, s:e] = 1

    return clean # type: ignore

@MEMORY.timed_cache(identifier_arg="file_path", ignore=["cleaned_binary", "original_normalized"])
def _convert_back(file_path: Path, cleaned_binary: np.ndarray, original_normalized: np.ndarray) -> TimedCacheResult[np.ndarray]:
    return cleaned_binary * original_normalized # type: ignore

class NormalizedPitchClassesCleaner:
    def __init__(self, logger: HoornLogger, n_fft: int = 2048, hop_length: int = 512):
        self._logger = logger
        self._separator = self.__class__.__name__

        self._n_fft = n_fft
        self._hop_length = hop_length

        self._logger.trace("Successfully initialized.", separator=self._separator)

    @staticmethod
    def _summarize_results(results: List[TimedCacheResult[Any]], final_binary: np.ndarray, final_chroma: np.ndarray) -> TimedCacheResult[Tuple[np.ndarray, np.ndarray]]:
        total_waiting: float = 0.0
        total_processing: float = 0.0

        for result in results:
            total_waiting += result.time_waiting
            total_processing += result.time_processing

        return TimedCacheResult(
            value=(final_binary, final_chroma),
            time_waiting=total_waiting,
            time_processing=total_processing,
            retrieved_from_cache=True
        )

    def clean(self, file_path: Path, normalized: np.ndarray, raw_audio_samples: np.ndarray, sample_rate: int) -> TimedCacheResult[Tuple[np.ndarray, np.ndarray]]:
        envelope_db = _compute_rms_envelope(file_path, raw_audio_samples, self._n_fft, self._hop_length)

        cleaned_pass1 = _first_clean_pass(file_path, envelope_db.value, normalized)
        perc_changed_pass1 = _get_perc_diff(file_path, normalized, cleaned_pass1.value)
        self._logger.trace(f"Cleaning (1st pass) changed {perc_changed_pass1.value:.4f}% frames!", separator=self._separator)
        self._logger.trace(f"Cleaned (1st pass):\n{pprint.pformat(cleaned_pass1.value)}", separator=self._separator)

        cleaned_pass2 = _second_clean_pass(file_path, cleaned_pass1.value)
        perc_changed_pass2 = _get_perc_diff(file_path, cleaned_pass1.value, cleaned_pass2.value)
        self._logger.trace(f"Cleaning (2nd pass) changed {perc_changed_pass2.value:.4f}% frames in comparison to first pass!", separator=self._separator)
        self._logger.trace(f"Cleaned (2nd pass):\n{pprint.pformat(cleaned_pass2.value)}", separator=self._separator)

        cleaned_pass3 = _third_clean_pass(file_path, cleaned_pass2.value)
        perc_changed_pass3 = _get_perc_diff(file_path, cleaned_pass2.value, cleaned_pass3.value)
        self._logger.trace(f"Cleaning (3rd pass) changed {perc_changed_pass3.value:.4f}% frames in comparison to second pass!", separator=self._separator)
        self._logger.trace(f"Cleaned (3rd pass):\n{pprint.pformat(cleaned_pass3.value)}", separator=self._separator)

        cleaned_pass4 = _fourth_clean_pass(file_path, cleaned_pass3.value)
        perc_changed_pass4 = _get_perc_diff(file_path, cleaned_pass3.value, cleaned_pass4.value)
        self._logger.trace(f"Cleaning (4th pass) changed {perc_changed_pass4.value:.4f}% frames in comparison to third pass!", separator=self._separator)
        self._logger.trace(f"Cleaned (4th pass):\n{pprint.pformat(cleaned_pass4.value)}", separator=self._separator)

        cleaned_pass5_binary = _fifth_clean_pass(file_path, cleaned_pass4.value, sample_rate, self._hop_length)
        perc_changed_pass5 = _get_perc_diff(file_path, cleaned_pass4.value, cleaned_pass5_binary.value)
        self._logger.trace(f"Cleaning (5th pass) changed {perc_changed_pass5.value:.4f}% frames in comparison to fourth pass!", separator=self._separator)
        self._logger.trace(f"Cleaned (5th pass):\n{pprint.pformat(cleaned_pass5_binary.value)}", separator=self._separator)

        cleaned_pass5_chroma = _convert_back(file_path, cleaned_pass5_binary.value, normalized)
        self._logger.trace(f"Cleaned (5th pass/normalized):\n{pprint.pformat(cleaned_pass5_chroma.value)}", separator=self._separator)

        return self._summarize_results(
            results=[
                envelope_db,
                cleaned_pass1, cleaned_pass2, cleaned_pass3, cleaned_pass4,
                cleaned_pass5_binary, cleaned_pass5_chroma
            ],
            final_binary=cleaned_pass5_binary.value,
            final_chroma=cleaned_pass5_chroma.value,
        )
