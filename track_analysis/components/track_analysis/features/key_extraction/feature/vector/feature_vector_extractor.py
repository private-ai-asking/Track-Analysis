from typing import List, Tuple

import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.key_extraction.feature.transforming.lof_feature_transformer import \
    LOFFeatureTransformer
from track_analysis.components.track_analysis.features.key_extraction.preprocessing.profiling.segment_profiler import Segment


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

    def extract(
            self,
            segments: List[Segment],
    ) -> Tuple[List[np.ndarray], List[Tuple[float,float]]]:
        self._logger.debug(f"Extracting features from {len(segments)} segments.", separator=self._separator)
        vectors: List[np.ndarray] = []
        intervals: List[Tuple[float, float]] = []
        for i, seg in enumerate(segments):
            hist = np.zeros(12, dtype=float)
            for event in seg.segment_notes:
                weight = event.note_duration_seconds_in_segment # * event.mean_energy_in_segment
                hist[event.pitch_class] += weight

            # Continue exactly as before—apply your LOF transformer, then normalize
            mapped = self._transformer.transform(hist)
            total = mapped.sum()
            norm = mapped / total if total > 0 else mapped

            vectors.append(norm)
            intervals.append((seg.segment_start_seconds, seg.segment_end_seconds))
            self._logger.trace(f"Segment {i}: extracted weighted feature vector.", separator=self._separator)

        self._logger.info("Completed feature extraction.", separator=self._separator)
        return vectors, intervals
