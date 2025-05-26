import pprint
from math import ceil
from typing import Callable, Dict, List, Optional

import numpy as np
from librosa.feature import chroma_stft

from skimage.filters import threshold_otsu

from track_analysis.components.md_common_python.py_common.logging import HoornLogger


class PathfindingHelper:
    """Utility class for key calculation and Viterbi pathfinding using Otsu's adaptive dB thresholding."""
    def __init__(
            self,
            logger: HoornLogger,
            log_idx: List[int],
            raw_key_score_calculator_func: Callable[[np.ndarray], Dict[str, float]],
            modulation_penalty_func: Optional[Callable[[str, str], float]] = None
    ):
        self._logger: HoornLogger = logger
        self._separator: str = "PathfindingHelper"

        # function scoring a binary-profile -> raw scores per key
        self._raw_key_score_calculator = raw_key_score_calculator_func
        # penalty function for key transitions
        self._penalty = modulation_penalty_func or (lambda prev, curr: 0.0 if prev == curr else 6.0)

        # Line-of-fifths index mapping
        self._lof_idx: List[int] = log_idx

        self._logger.trace("Initialized PathfindingHelper with Otsu thresholding.", separator=self._separator)

    def _compute_binary_profile(
            self,
            chroma: np.ndarray,
            segment_idx: int
    ) -> np.ndarray:
        """
        Convert a 12-bin raw chroma vector into a binary LOF-ordered profile
        using Otsu's method on per-segment dB values.
        """
        # 1) Convert to decibels relative to segment max
        eps = 1e-9
        segment_max = chroma.max() or 1.0
        decibels = 20 * np.log10(chroma / (segment_max + eps) + eps)

        print(f"[Segment {segment_idx}] In dB (0 dB = segment max):")
        pprint.pprint(decibels)

        thresh_db = threshold_otsu(decibels)

        print(f"[Segment {segment_idx}] Otsu threshold: {thresh_db:.2f} dB")

        # 3) Binary presence
        binary = (decibels >= thresh_db).astype(np.int8)
        print(f"[Segment {segment_idx}] Binary presence (>=Otsu):")
        pprint.pprint(binary)

        # 4) Reorder into LOF space
        return binary[self._lof_idx]

    def _score_profile(
            self,
            binary_profile: np.ndarray,
            segment_idx: int
    ) -> Dict[str, float]:
        scores = self._raw_key_score_calculator(binary_profile)
        print(f"[Segment {segment_idx}] Raw key-score:")
        pprint.pprint(scores)
        best = max(scores, key=scores.get)
        print(f"[Segment {segment_idx}] Best key {best} ({scores[best]:.3f})")
        # exit()
        return scores

    def process(
            self,
            segments: List[np.ndarray],
            sample_rate: int,
            frames_per_segment_window: int
    ) -> List[str]:
        if not segments:
            return []

        # Precompute mean-chroma per segment
        chromas = []
        for idx, seg in enumerate(segments, start=1):
            hop = ceil(seg.shape[0] / frames_per_segment_window)
            C = chroma_stft(y=seg, sr=sample_rate, n_fft=1024, hop_length=hop, norm=None)
            print(f"[Segment {idx}] STFT frames:")
            pprint.pprint(C)
            m = C.mean(axis=1)
            print(f"[Segment {idx}] Mean energy per bin:")
            pprint.pprint(m)
            chromas.append(m)

        # Compute raw scores
        raw_scores = []
        for idx, chi in enumerate(chromas, start=1):
            binary = self._compute_binary_profile(chi, idx)
            raw_scores.append(self._score_profile(binary, idx))
            self._logger.info(f"Processed {idx}/{len(chromas)} segments", separator=self._separator)

        # Viterbi
        keys = list(raw_scores[0].keys())
        N, K = len(raw_scores), len(keys)
        C = np.full((N, K), -np.inf)
        P = np.zeros((N, K), dtype=int)
        for i, k in enumerate(keys):
            C[0, i] = raw_scores[0][k]
        for t in range(1, N):
            for i, curr in enumerate(keys):
                best_val, best_j = -np.inf, 0
                for j, prev in enumerate(keys):
                    v = C[t-1, j] - self._penalty(prev, curr) + raw_scores[t][curr]
                    if v > best_val:
                        best_val, best_j = v, j
                C[t, i], P[t, i] = best_val, best_j
        # backtrack
        path = [None] * N
        j = int(np.argmax(C[-1]))
        for t in range(N-1, -1, -1):
            path[t] = keys[j]
            j = P[t, j]
        self._logger.info("Completed key pathfinding.", separator=self._separator)
        return path
