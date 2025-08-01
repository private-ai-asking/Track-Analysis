from pathlib import Path
from typing import List, Tuple

import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import VERBOSE
from track_analysis.components.track_analysis.features.core.cacheing.onset_envelope import OnsetStrengthExtractor
from track_analysis.components.track_analysis.features.key_extraction.preprocessing.segmentation.model.segmentation_result import \
    SegmentationResult


class SegmentSlicer:
    """
    Slices audio segments at specified event boundaries.
    """
    _INITIAL_BOUNDARY_FACTOR = 2

    def __init__(self, logger: HoornLogger, separator: str) -> None:
        self._logger = logger
        self._separator = separator
        self._onset_extractor: OnsetStrengthExtractor = OnsetStrengthExtractor(logger)

    def slice_segments(
            self,
            audio: np.ndarray,
            audio_path: Path,
            percussive: np.ndarray,
            sample_rate: int,
            event_times: List[float],
            strong_times: List[float],
            hop_length_samples: int,
    ) -> SegmentationResult:
        """
        Slice the provided audio into segments based on filtered strong time boundaries.
        """
        self._validate_inputs(audio, sample_rate, event_times)

        onset_strength_envelope = self._onset_extractor.extract(
            file_path=audio_path,
            start_sample=0,
            end_sample=percussive.shape[0],
            sample_rate=sample_rate,
            hop_length=hop_length_samples,
            audio=percussive,
        )

        boundaries = self._filter_boundaries(event_times[0], strong_times)
        segments, start_times, durations = self._extract_segments(
            audio, sample_rate, event_times[0], boundaries
        )
        self._log(durations)
        return SegmentationResult(
            segments, start_times, durations, onset_strength_envelope
        )

    @staticmethod
    def _validate_inputs(
            audio: np.ndarray,
            sample_rate: int,
            event_times: List[float],
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        if not audio.size:
            raise ValueError("audio array cannot be empty")
        if not event_times:
            raise ValueError("event_times cannot be empty")

    def _filter_boundaries(
            self,
            start_time: float,
            strong_times: List[float],
    ) -> List[float]:
        """
        Filter out initial strong times too close to the first event boundary.
        """
        filtered: List[float] = []
        for idx, boundary in enumerate(strong_times):
            if idx == 0 and len(strong_times) > 1:
                next_gap = strong_times[1] - start_time
                if (boundary - start_time) < next_gap / self._INITIAL_BOUNDARY_FACTOR:
                    continue
            filtered.append(boundary)
        return filtered

    @staticmethod
    def _extract_segments(
            audio: np.ndarray,
            sample_rate: int,
            start_time: float,
            boundaries: List[float],
    ) -> Tuple[List[np.ndarray], List[float], List[float]]:
        segments: List[np.ndarray] = []
        start_times: List[float] = [start_time]
        durations: List[float] = []
        previous = start_time

        for boundary in boundaries:
            start_idx = int(previous * sample_rate)  # Previous end in seconds * sample rate (samples/second) = current start sample.
            end_idx = int(boundary * sample_rate)
            segments.append(audio[start_idx:end_idx])
            durations.append(boundary - previous)
            start_times.append(boundary)
            previous = boundary

        final_time = audio.shape[0] / sample_rate
        if previous < final_time:
            segments.append(audio[int(previous * sample_rate):])
            durations.append(final_time - previous)

        return segments, start_times, durations

    def _log(self, durations: List[float]) -> None:
        if VERBOSE:
            self._logger.debug(
                f"Segment durations: {durations}", separator=self._separator
            )
