from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.preprocessing.segmentation.metrical_hierarchy_constructor import \
    MetricalHierarchyConstructor
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.preprocessing.segmentation.model.segmentation_result import \
    SegmentationResult
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.preprocessing.segmentation.segment_slicer import \
    SegmentSlicer


@dataclass(frozen=True)
class RawSegment:
    samples: np.ndarray
    segment_start_seconds: float
    segment_end_seconds: float
    segment_duration_seconds: float


class AudioSegmenter:
    """
    Coordinates beat detection, hierarchy assignment, and slicing.
    """
    def __init__(
            self,
            logger: HoornLogger,
            hop_length_samples: int = 512,
            subdivisions_per_beat: int = 2,
            beats_per_segment: int = 8
    ):
        self._logger = logger
        self._separator = "AudioSegmenter"
        self._beats_per_segment: int = beats_per_segment
        self._hierarchy_constructor = MetricalHierarchyConstructor(subdivisions_per_beat, logger, self._separator)
        self._slicer = SegmentSlicer(logger, self._separator)

        self._hop_length_samples = hop_length_samples

        self._logger.trace(
            "Initialized AudioSegmenter",
            separator=self._separator,
        )
        self._logger.debug(
            f"Configuration: subdivisions_per_beat={subdivisions_per_beat}",
            separator=self._separator,
        )

    def get_segments(
            self,
            audio_path: Path,
            beat_frames: np.ndarray,
            beat_times: np.ndarray,
            audio_samples: np.ndarray,
            sample_rate: int,
            min_segment_level: int = 3,
    ) -> List[RawSegment]:
        self._logger.debug(
            f"Segmenting audio ({len(audio_samples)} samples at {sample_rate}Hz) "
            f"with {self._beats_per_segment} beats per segment",
            separator=self._separator,
        )

        if len(beat_frames) < self._beats_per_segment:
            self._logger.warning(
                "Too few beats for a full segment; cannot segment.",
                separator=self._separator,
            )
            return []

        level_array, event_times = self._hierarchy_constructor.construct_hierarchy(
            audio_path, beat_times, beat_frames, self._beats_per_segment
        )
        strong_times = [t for t, lvl in zip(event_times, level_array) if lvl >= min_segment_level]

        segmentation_result: SegmentationResult = self._slicer.slice_segments(
            audio_samples, sample_rate, event_times, strong_times
        )

        segments: List[RawSegment] = []

        for segments_samples, start_time, duration in zip(segmentation_result.segments, segmentation_result.start_times, segmentation_result.durations):
            segments.append(RawSegment(
                samples=audio_samples,
                segment_start_seconds=start_time,
                segment_end_seconds=start_time+duration,
                segment_duration_seconds=duration
            ))

        return segments
