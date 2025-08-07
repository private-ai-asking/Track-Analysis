from typing import List

import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.feature.lof.lof_feature_transformer import \
    LOFFeatureTransformer
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.preprocessing.note_extraction.note_event_builder import \
    NoteEvent

from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.preprocessing.profiling.segment_profiler import \
    ProfiledSegment, SegmentNote


class FeatureVectorExtractor:
    """
    Extracts normalized LOF-based feature vectors and their time intervals from segments.
    """
    def __init__(
            self,
            logger: HoornLogger,
            transformer: LOFFeatureTransformer
    ):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._transformer = transformer
        self._logger.debug("Initialized FeatureVectorExtractor.", separator=self._separator)

    def extract_segments(
            self,
            segments: List[ProfiledSegment],
    ) -> List[np.ndarray]:
        self._logger.debug(f"Extracting features from {len(segments)} segments.", separator=self._separator)
        vectors: List[np.ndarray] = []
        for i, seg in enumerate(segments):
            segment_feature_vector = self._extract_segment(seg)
            vectors.append(segment_feature_vector)
            self._logger.trace(f"Segment {i}: extracted weighted feature vector.", separator=self._separator)

        self._logger.info("Completed feature extraction.", separator=self._separator)
        return vectors

    def extract_features_from_note_events(self, events: List[NoteEvent]) -> np.ndarray:
        hist = np.zeros(12, dtype=float)

        for event in events:
            hist[event.pitch_class] += self._get_note_score(event)

        return hist

    def _extract_segment(self, segment: ProfiledSegment) -> np.ndarray:
        hist = np.zeros(12, dtype=float)
        for event in segment.segment_notes:
            hist[event.pitch_class] += self._get_note_score(event)

        mapped = self._transformer.transform(hist)
        total = mapped.sum()
        norm = mapped / total if total > 0 else mapped
        return norm

    @staticmethod
    def _get_note_score(note: SegmentNote | NoteEvent) -> float:
        if isinstance(note, NoteEvent):
            return note.duration_seconds * note.total_energy
        else:
            return note.note_duration_seconds_in_segment * note.total_energy_in_segment
