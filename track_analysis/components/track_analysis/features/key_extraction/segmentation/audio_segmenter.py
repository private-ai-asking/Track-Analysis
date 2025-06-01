from pathlib import Path
from typing import Tuple, Optional

import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.key_extraction.segmentation.beat_detector import BeatDetector
from track_analysis.components.track_analysis.features.key_extraction.segmentation.metrical_hierarchy_constructor import \
    MetricalHierarchyConstructor
from track_analysis.components.track_analysis.features.key_extraction.segmentation.model.segmentation_result import \
    SegmentationResult
from track_analysis.components.track_analysis.features.key_extraction.segmentation.segment_slicer import SegmentSlicer


class AudioSegmenter:
    """
    Coordinates beat detection, hierarchy assignment, and slicing.
    """
    def __init__(
            self,
            logger: HoornLogger,
            cache_dir: Path,
            hop_length_samples: int = 512,
            subdivisions_per_beat: int = 2,
            beats_per_segment: int = 8
    ):
        self._logger = logger
        self._separator = "AudioSegmenter"
        self._beats_per_segment: int = beats_per_segment
        self._detector = BeatDetector(logger, self._separator, cache_dir / "beat detection")
        self._hierarchy = MetricalHierarchyConstructor(subdivisions_per_beat, logger, self._separator, cache_dir / "hierarchy construction")
        self._slicer = SegmentSlicer(logger, self._separator, cache_dir / "slicing")

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
            audio_samples: np.ndarray,
            sample_rate: int,
            min_segment_level: int = 3,
    ) -> Optional[Tuple[SegmentationResult, float]]:
        self._logger.debug(
            f"Segmenting audio ({len(audio_samples)} samples at {sample_rate}Hz) "
            f"with {self._beats_per_segment} beats per segment",
            separator=self._separator,
        )

        tempo, frames, times = self._detector.detect(audio_samples, sample_rate)
        if len(frames) < self._beats_per_segment:
            self._logger.warning(
                "Too few beats for a full segment; cannot segment.",
                separator=self._separator,
            )
            return None

        level_array, event_times = self._hierarchy.construct_hierarchy(
            times, frames, self._beats_per_segment
        )
        strong_times = [t for t, lvl in zip(event_times, level_array) if lvl >= min_segment_level]

        return self._slicer.slice_segments(
            audio_samples, sample_rate, event_times, strong_times, self._hop_length_samples
        ), tempo
