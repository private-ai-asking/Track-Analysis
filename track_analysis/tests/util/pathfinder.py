import pprint
from math import ceil
from typing import Callable, Dict, List, Optional, Literal

import numpy as np
from librosa.feature import chroma_stft
from scipy.ndimage import median_filter

from track_analysis.components.md_common_python.py_common.logging import HoornLogger


class PathfindingHelper:
    """Utility class for key calculation pathfinding helping."""
    def __init__(
            self,
            logger: HoornLogger,
            raw_key_score_calculator_func: Callable[[np.ndarray, Literal], Dict[str, float]],
            modulation_penalty_func: Optional[Callable[[str, str], float]] = None
    ):
        self._logger: HoornLogger = logger
        self._separator: str = "PathfindingHelper"

        # function scoring a flat-profile -> raw scores per key
        self._raw_key_score_calculator = raw_key_score_calculator_func
        # penalty function for key transitions
        if modulation_penalty_func is None:
            self._penalty = lambda prev, curr: 0.0 if prev == curr else 6.0
        else:
            self._penalty = modulation_penalty_func

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def _get_raw_scores_for_segment(
            self,
            segment: np.ndarray,
            sample_rate: int,
            frames_per_segment_window: int
    ) -> Dict[str, float]:
        # 1) Compute hop_length from desired number of frames
        segment_sample_count = segment.shape[0]
        hop_length = ceil(segment_sample_count / frames_per_segment_window)

        # 2) Compute chroma
        chroma = chroma_stft(
            y=segment,
            sr=sample_rate,
            n_fft=1024,
            hop_length=hop_length,
            norm=2
        )

        chroma = median_filter(chroma, size=(1, 3))

        # print("Original:")
        # pprint.pprint(chroma)

        # 3) Average over time to get 12-bin profile
        chroma = chroma.mean(axis=1)

        # print("Average:")
        # pprint.pprint(chroma)

        # 4) Adaptive thresholding via percentile
        # thr = np.percentile(chroma, 75)         # choose any percentile you like
        # chroma = (chroma > thr).astype(np.int8)

        chroma = chroma / (chroma.sum() + 1e-8)

        # 5) Score against your (weighted) templates
        scores: Dict[str, float] = self._raw_key_score_calculator(chroma, mode="average")  # type: ignore

        # print("Raw key-score:")
        # pprint.pprint(scores)

        return scores

    def process(
            self,
            segments: List[np.ndarray],
            sample_rate: int,
            frames_per_segment_window: int
    ) -> List[str]:
        """
        Perform Viterbi-style cumulative_key_scores over all segments to find the optimal key sequence.

        Returns:
            A list of predicted keys, one per segment.
        """
        # 1) Gather raw scores per segment
        raw_scores: List[Dict[str, float]] = []
        total_segments: int = len(segments)
        processed: int = 0
        for segment in segments:
            scores = self._get_raw_scores_for_segment(
                segment,
                sample_rate,
                frames_per_segment_window
            )
            raw_scores.append(scores)
            processed += 1
            self._logger.info(
                f"Processed {processed}/{total_segments} ({processed/total_segments*100:.2f}%) segments.",
                separator=self._separator
            )

        if not raw_scores:
            return []

        self._logger.info("Starting the backtracking process...", separator=self._separator)

        # 2) Prepare cumulative_key_scores tables
        keys = list(raw_scores[0].keys())
        num_segments = len(raw_scores)
        num_states = len(keys)

        cumulative_key_scores = np.full((num_segments, num_states), -np.inf, dtype=float)
        predecessor_state = np.zeros((num_segments, num_states), dtype=int)

        # 3) Initialization for first segment
        for i, key in enumerate(keys):
            cumulative_key_scores[0, i] = raw_scores[0][key]
            predecessor_state[0, i] = i

        # 4) Recurrence
        for t in range(1, num_segments):
            for i, curr_key in enumerate(keys):
                score_t = raw_scores[t][curr_key]
                best_prev_score = -np.inf
                best_prev_index = 0
                for j, prev_key in enumerate(keys):
                    penalty = self._penalty(prev_key, curr_key)
                    candidate = cumulative_key_scores[t-1, j] - penalty + score_t
                    if candidate > best_prev_score:
                        best_prev_score = candidate
                        best_prev_index = j
                cumulative_key_scores[t, i] = best_prev_score
                predecessor_state[t, i] = best_prev_index

        # 5) Termination: pick best final state
        final_index = int(np.argmax(cumulative_key_scores[-1, :]))

        # 6) Backtracking to recover path
        best_path: List[str] = [None] * num_segments  # type: ignore
        best_path[-1] = keys[final_index]
        for t in range(num_segments-1, 0, -1):
            prev_index = predecessor_state[t, final_index]
            best_path[t-1] = keys[prev_index]
            final_index = prev_index

        self._logger.info("Done backtracking; calculated best path.", separator=self._separator)
        return best_path
