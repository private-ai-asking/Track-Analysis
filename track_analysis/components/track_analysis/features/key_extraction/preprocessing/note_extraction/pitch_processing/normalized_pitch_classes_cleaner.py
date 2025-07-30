import os
import pprint
from pathlib import Path

import librosa
import numpy as np
from joblib import Memory
from librosa.feature import rms
from typing import Tuple
from scipy.signal import medfilt
from skimage.morphology import binary_opening, binary_closing

from track_analysis.components.md_common_python.py_common.logging import HoornLogger

def _compute_rms_envelope(
        data: np.ndarray,
        n_fft: int,
        hop_length: int,
) -> np.ndarray:
    result = rms(
        y=data,
        frame_length=n_fft,
        hop_length=hop_length,
        center=True
    )[0]                       # shape = (n_frames,)

    envelope_db = librosa.amplitude_to_db(result, ref=np.max)
    return envelope_db

def _get_perc_diff(a: np.ndarray, b: np.ndarray) -> float:
    diffs = ~np.isclose(a, b, rtol=1e-5, atol=1e-8)
    n_changed = np.count_nonzero(diffs)
    perc_changed = 100 * (n_changed / a.size)
    return perc_changed

def _first_clean_pass(envelope_db: np.ndarray, normalized: np.ndarray) -> np.ndarray:
    mask = (envelope_db >= -60)  # boolean (T=above noise-floor)
    cleaned_pass1 = normalized * mask[np.newaxis, :]
    return cleaned_pass1

def _second_clean_pass(cleaned_first_pass: np.ndarray) -> np.ndarray:
    return medfilt(cleaned_first_pass, kernel_size=(1, 5))

def _third_clean_pass(cleaned_second_pass: np.ndarray, threshold_db: int = -20) -> np.ndarray:
    cleaned_second_pass_db = librosa.amplitude_to_db(cleaned_second_pass, ref=np.max)
    return (cleaned_second_pass_db > threshold_db).astype(np.uint8)

def _fourth_clean_pass(cleaned_third_pass: np.ndarray) -> np.ndarray:
    # make sure we’re working with booleans
    mask = cleaned_third_pass.astype(bool)

    # 1×3 structuring element
    footprint = np.ones((1, 5), dtype=bool)

    # opening to remove tiny spikes, then closing to fill tiny holes
    opened = binary_opening(mask, footprint=footprint)
    closed = binary_closing(opened, footprint=footprint)

    # return as the same dtype you expect downstream
    return closed.astype(np.uint8)

def _fifth_clean_pass(cleaned_fourth_pass: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    min_len = int(0.075 * sr / hop_length)
    clean = np.zeros_like(cleaned_fourth_pass)
    for i in range(cleaned_fourth_pass.shape[0]):
        on_off = np.diff(np.concatenate([[0], cleaned_fourth_pass[i], [0]]))  # type: ignore
        starts = np.where(on_off==1)[0]
        ends   = np.where(on_off==-1)[0]
        for s,e in zip(starts, ends):
            if (e-s) >= min_len:
                clean[i, s:e] = 1

    return clean

def _convert_back(cleaned_binary: np.ndarray, original_normalized: np.ndarray) -> np.ndarray:
    return cleaned_binary * original_normalized

class NormalizedPitchClassesCleaner:
    def __init__(self, logger: HoornLogger, cache_dir: Path):
        self._logger = logger
        self._separator = self.__class__.__name__

        os.makedirs(cache_dir, exist_ok=True)
        self._compute_rms = Memory(cache_dir, verbose=0).cache(_compute_rms_envelope)
        self._get_perc_diff = Memory(cache_dir, verbose=0).cache(_get_perc_diff)
        self._clean_pass1 = Memory(cache_dir, verbose=0).cache(_first_clean_pass)
        self._clean_pass2 = Memory(cache_dir, verbose=0).cache(_second_clean_pass)
        self._clean_pass3 = Memory(cache_dir, verbose=0).cache(_third_clean_pass)
        self._clean_pass4 = Memory(cache_dir, verbose=0).cache(_fourth_clean_pass)
        self._clean_pass5 = Memory(cache_dir, verbose=0).cache(_fifth_clean_pass)
        self._convert_back = Memory(cache_dir, verbose=0).cache(_convert_back)

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def clean(self, normalized: np.ndarray, raw_audio_samples: np.ndarray, n_fft: int, hop_length: int, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        envelope_db = self._compute_rms(raw_audio_samples, n_fft, hop_length)

        cleaned_pass1 = self._clean_pass1(envelope_db, normalized)
        perc_changed_pass1 = _get_perc_diff(normalized, cleaned_pass1)
        self._logger.debug(f"Cleaning (1st pass) changed {perc_changed_pass1:.4f}% frames!", separator=self._separator)
        self._logger.debug(f"Cleaned (1st pass):\n{pprint.pformat(cleaned_pass1)}", separator=self._separator)

        cleaned_pass2 = self._clean_pass2(cleaned_pass1)
        perc_changed_pass2 = self._get_perc_diff(cleaned_pass1, cleaned_pass2)
        self._logger.debug(f"Cleaning (2nd pass) changed {perc_changed_pass2:.4f}% frames in comparison to first pass!", separator=self._separator)
        self._logger.debug(f"Cleaned (2nd pass):\n{pprint.pformat(cleaned_pass2)}", separator=self._separator)

        cleaned_pass3 = self._clean_pass3(cleaned_pass2)
        perc_changed_pass3 = self._get_perc_diff(cleaned_pass2, cleaned_pass3)
        self._logger.debug(f"Cleaning (3rd pass) changed {perc_changed_pass3:.4f}% frames in comparison to second pass!", separator=self._separator)
        self._logger.debug(f"Cleaned (3rd pass):\n{pprint.pformat(cleaned_pass3)}", separator=self._separator)

        cleaned_pass4 = self._clean_pass4(cleaned_pass3)
        perc_changed_pass4 = self._get_perc_diff(cleaned_pass3, cleaned_pass4)
        self._logger.debug(f"Cleaning (4th pass) changed {perc_changed_pass4:.4f}% frames in comparison to third pass!", separator=self._separator)
        self._logger.debug(f"Cleaned (4th pass):\n{pprint.pformat(cleaned_pass4)}", separator=self._separator)

        cleaned_pass5_binary = self._clean_pass5(cleaned_pass4, sample_rate, hop_length)
        perc_changed_pass5 = self._get_perc_diff(cleaned_pass4, cleaned_pass5_binary)
        self._logger.debug(f"Cleaning (5th pass) changed {perc_changed_pass5:.4f}% frames in comparison to fourth pass!", separator=self._separator)
        self._logger.debug(f"Cleaned (5th pass):\n{pprint.pformat(cleaned_pass5_binary)}", separator=self._separator)

        cleaned_pass5_chroma = self._convert_back(cleaned_pass5_binary, normalized)
        self._logger.debug(f"Cleaned (5th pass/normalized):\n{pprint.pformat(cleaned_pass5_chroma)}", separator=self._separator)

        return cleaned_pass5_binary, cleaned_pass5_chroma
