from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.testing import TestInterface
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.note_extractor import \
    NoteExtractor
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.segment_profiler import \
    SegmentProfiler, Segment


def lof_mapping(arr: np.ndarray) -> np.ndarray:
    # chromatic->LOF index map
    idx = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    return arr[idx]


class KeyProgressionTest(TestInterface):
    def __init__(self, logger: HoornLogger, modulation_penalty: float = 6.0):
        super().__init__(logger, is_child=True)
        self._separator = 'KeyProgressionTest'

        # base profiles in chromatic order
        ionian = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
        aeolian = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])
        dorian = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
        modes = {'Ionian (Major)': ionian, 'Aeolian (Minor)': aeolian, 'Dorian (Minor)': dorian}
        lof_tonics = ['C','G','D','A','E','B','F#','C#','G#','D#','A#','F']

        # build LOF templates
        self._templates: Dict[str, np.ndarray] = {}
        for mode_name, arr in modes.items():
            base = lof_mapping(arr)
            for shift, tonic in enumerate(lof_tonics):
                name = f"{tonic} {mode_name}"
                tmpl = np.roll(base, shift)
                self._templates[name] = tmpl / tmpl.sum()

        self._penalty = modulation_penalty * 2.0 / ionian.sum()
        self._logger.debug(
            f"Penalty: (original={modulation_penalty};new={self._penalty:0.4f})",
            separator=self._separator
        )

        self._note_extractor = NoteExtractor(logger, subdivisions_per_beat=2, hop_length_samples=512)
        self._segment_profiler = SegmentProfiler(logger)

    def test(
            self,
            file_path: Path,
            time_signature: Tuple[int, int] = (4, 4),
            segment_beat_level: int = 3
    ) -> None:
        if not file_path.is_file():
            self._logger.error(f"File not found: {file_path}", separator=self._separator)
            return
        self._logger.info(f"Analyzing file: {file_path}", separator=self._separator)

        note_events, seg_res = self._note_extractor.extract(
            file_path, time_signature, segment_beat_level, visualize=False
        )
        segments = self._segment_profiler.profile_segments(seg_res, note_events)

        keys = list(self._templates.keys())
        raw_scores = self._compute_raw_scores(segments, keys)
        path = self._compute_dp_path(raw_scores)

        self._log_individual(segments, keys, raw_scores, path)
        self._log_merged(segments, keys, path)
        self._logger.info("Key progression complete.", separator=self._separator)

    def _compute_raw_scores(
            self,
            segments: List,
            keys: List[str]
    ) -> np.ndarray:
        n = len(segments)
        m = len(keys)
        raw_scores = np.zeros((n, m))
        for i, seg in enumerate(segments):
            hist = np.zeros(12)
            for e in seg.segment_notes:
                hist[e.pitch_class] += 1
            lof_hist = lof_mapping(hist)
            total = lof_hist.sum()
            pc_norm = lof_hist / total if total > 0 else lof_hist
            for j, kname in enumerate(keys):
                raw_scores[i, j] = np.corrcoef(pc_norm, self._templates[kname])[0, 1]
        return raw_scores

    def _compute_dp_path(self, raw_scores: np.ndarray) -> np.ndarray:
        n, m = raw_scores.shape
        dp = np.full((n, m), -np.inf)
        backptr = np.zeros((n, m), dtype=int)
        dp[0, :] = raw_scores[0, :]
        for i in range(1, n):
            for j in range(m):
                stay = dp[i - 1, j]
                mod_vals = dp[i - 1, :] - self._penalty
                mod = np.max(mod_vals)
                best_prev = j if stay >= mod else int(np.argmax(mod_vals))
                dp[i, j] = raw_scores[i, j] + max(stay, mod)
                backptr[i, j] = best_prev
        path = np.zeros(n, dtype=int)
        path[-1] = int(np.argmax(dp[-1, :]))
        for i in range(n - 2, -1, -1):
            path[i] = backptr[i + 1, path[i + 1]]
        return path

    def _log_individual(
            self,
            segments: List,
            keys: List[str],
            raw_scores: np.ndarray,
            path: np.ndarray
    ) -> None:
        for i, seg in enumerate(segments):
            key = keys[path[i]]
            score = raw_scores[i, path[i]]
            self._logger.debug(
                f"Segment {i}: Key={key}, Score={score:.3f}",
                separator=self._separator
            )

    def _log_merged(
            self,
            segments: List[Segment],
            keys: List[str],
            path: np.ndarray
    ) -> None:
        merged = []  # type: List[Dict]
        start_idx = 0
        current_key = keys[path[0]]
        for i in range(1, len(path)):
            if keys[path[i]] != current_key:
                merged.append({
                    'start': segments[start_idx].segment_start_seconds,
                    'end': segments[i - 1].segment_end_seconds,
                    'key': current_key,
                    'idx': start_idx
                })
                start_idx = i
                current_key = keys[path[i]]
        # append last
        merged.append({
            'start': segments[start_idx].segment_start_seconds,
            'end': segments[-1].segment_end_seconds,
            'key': current_key,
            'idx': start_idx
        })
        # log
        for seg in merged:
            duration = seg['end'] - seg['start']
            self._logger.info(
                f"[segment {seg['idx']}] {self._format_time(seg['start'])} -> {self._format_time(seg['end'])} "
                f"({duration:.2f}s) => {seg['key']}",
                separator=self._separator
            )

    @staticmethod
    def _format_time(seconds: float) -> str:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}:{secs:05.2f}"
