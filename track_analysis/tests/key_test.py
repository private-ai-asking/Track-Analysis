import pprint
from pathlib import Path
from typing import Tuple, List, Dict, Literal

import librosa
import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.testing import TestInterface
from track_analysis.tests.util.pathfinder import PathfindingHelper


class KeyProgressionTest(TestInterface):
    def __init__(self, logger: HoornLogger, modulation_penalty: float = 6.0):
        super().__init__(logger, is_child=True)
        self._separator: str = 'KeyProgressionTest'

        # Temperley‑revised Ionian, Aeolian & Dorian profiles in chromatic order
        # C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B
        ionian = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
        aeolian = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])
        dorian  = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])

        # 1) Define Line‑of‑Fifths index mapping (from chromatic to LOF order)
        lof_idx = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

        # 2) Reorder each profile into LOF order
        ionian_lof  = ionian[lof_idx]
        aeolian_lof = aeolian[lof_idx]
        dorian_lof  = dorian[lof_idx]

        # 3) Define the 12 tonics in LOF order
        lof_tonics = ["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"]

        # 4) Map mode names to their LOF‑ordered, centered profiles
        modes_lof: Dict[str, np.ndarray] = {
            "Ionian (Major)": ionian_lof,
            "Aeolian (Minor)": aeolian_lof,
            "Dorian (Minor)":  dorian_lof,
        }

        # 5) Build templates by rotating each LOF profile by perfect‑fifth steps
        self._templates: Dict[str, np.ndarray] = {
            f"{tonic} {mode}": np.roll(profile, shift)
            for mode, profile in modes_lof.items()
            for shift, tonic in enumerate(lof_tonics)
        }

        self._pathfinder_helper = PathfindingHelper(
            logger,
            lof_idx,
            self._compute_profile_scores_for_segment,
            lambda prev, curr: 0.0 if prev == curr else modulation_penalty
        )


    def test(self, file_path: Path, tempo_bpm: float, time_signature: Tuple[int,int] = (4,4)) -> None:
        if not file_path.is_file():
            self._logger.error(f"File not found: {file_path}", separator=self._separator)
            return
        self._logger.info(f"Analyzing file: {file_path}", separator=self._separator)

        beats_per_bar = time_signature[0]
        frames_per_segment = beats_per_bar  # one frame per beat

        # Segment audio and get segment start times
        segments, start_times = self._segment_samples(samples_and_times := self._get_samples_and_sample_rate(file_path), time_signature)
        samples, sample_rate = samples_and_times

        # Find best key path
        best_path = self._pathfinder_helper.process(
            segments,
            sample_rate,
            frames_per_segment
        )

        # Merge contiguous segments with same key and compute their time spans
        merged = []  # List of tuples (key, start_time, end_time)
        current_key = best_path[0]
        seg_start = start_times[0]
        for idx, key in enumerate(best_path[1:], start=1):
            if key != current_key:
                # end of current merged segment
                seg_end = start_times[idx]
                merged.append((current_key, seg_start, seg_end))
                current_key = key
                seg_start = start_times[idx]
        # append the last segment
        last_idx = len(best_path) - 1
        seg_end = start_times[last_idx] + len(segments[last_idx]) / sample_rate
        merged.append((current_key, seg_start, seg_end))

        # Print merged segments with times
        for key, start, end in merged:
            self._logger.info(
                f"{self._format_time(start)} - {self._format_time(end)}: {key}",
                separator=self._separator
            )

        self._logger.info("Done", separator=self._separator)

    def _compute_profile_scores_for_segment(
            self,
            profile: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute cosine-similarity scores between the chroma profile and each key template.
        score = (profile · template) / (||profile|| * ||template||)
        """
        scores: Dict[str, float] = {}

        for key, tpl in self._templates.items():
            scores[key] = profile.dot(tpl)

        return scores

    def _segment_samples(
            self,
            samples_and_rate: Tuple[np.ndarray, int],
            time_signature: Tuple[int,int]
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Detect beats and slice audio into bar‐aligned segments, returning start times in seconds"""
        samples, sample_rate = samples_and_rate

        # 1) Beat tracking with initial tempo guess
        tempo_bpm, beat_frames = librosa.beat.beat_track(
            y=samples,
            sr=sample_rate,
            start_bpm=120,
            units='frames'
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
        beat_samples = (beat_times * sample_rate).astype(int)

        # 2) Group into bars
        beats_per_bar = time_signature[0]
        if len(beat_samples) < beats_per_bar:
            return [samples], [0.0]
        bar_sample_starts = beat_samples[::beats_per_bar]
        bar_time_starts = beat_times[::beats_per_bar]
        # include last frame boundary
        if bar_sample_starts[-1] != beat_samples[-1]:
            bar_sample_starts = np.append(bar_sample_starts, beat_samples[-1])
            bar_time_starts = np.append(bar_time_starts, samples.shape[0] / sample_rate)

        segments = [
            samples[bar_sample_starts[i]:bar_sample_starts[i+1]]
            for i in range(len(bar_sample_starts)-1)
        ]
        start_times = bar_time_starts[:-1].tolist()

        self._logger.debug(
            f"Segmented audio into {len(segments)} bar-aligned segments",
            separator=self._separator
        )
        return segments, start_times

    @staticmethod
    def _get_samples_and_sample_rate(file_path: Path) -> Tuple[np.ndarray, int]:
        samples, sample_rate = librosa.load(file_path, sr=None)
        return samples, round(sample_rate)

    @staticmethod
    def _format_time(seconds: float) -> str:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}:{secs:05.2f}"
