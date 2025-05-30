from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from track_analysis.components.md_common_python.py_common.algorithms.dynamic_programming import StateSequenceDecoder
from track_analysis.components.md_common_python.py_common.algorithms.sequence import RunLengthMerger
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.key_extraction.key_extraction.feature_vector_extractor import \
    FeatureVectorExtractor
from track_analysis.components.track_analysis.features.key_extraction.key_extraction.key_template_builder import \
    KeyTemplateBuilder
from track_analysis.components.md_common_python.py_common.algorithms.similarity import SimilarityMatcher
from track_analysis.components.track_analysis.features.key_extraction.key_extraction.lof_feature_transformer import \
    LOFFeatureTransformer
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.note_extractor import \
    NoteExtractor
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.segment_profiler import \
    SegmentProfiler


class KeyProgressionAnalyzer:
    """
    Orchestrates key progression analysis: extraction, matching, decoding, merging.
    """
    def __init__(
            self,
            logger: HoornLogger,
            modulation_penalty: float = 6.0,
            modes: Dict[str, np.ndarray] = None,
            tonics: List[str] = None
    ):
        self._logger = logger
        self._separator = self.__class__.__name__
        # default music modes and tonics
        self._modes = modes or {
            'Ionian (Major)': np.array([5.0,2.0,3.5,2.0,4.5,4.0,2.0,4.5,2.0,3.5,1.5,4.0]),
            'Aeolian (Minor)': np.array([5.0,2.0,3.5,4.5,2.0,4.0,2.0,4.5,3.5,2.0,1.5,4.0]),
            'Dorian (Minor)': np.array([5.0,2.0,3.5,4.5,2.0,4.0,2.0,4.5,2.0,3.5,1.5,4.0])
        }
        self._tonics = tonics or ['C','G','D','A','E','B','F#','C#','G#','D#','A#','F']

        builder = KeyTemplateBuilder(logger, self._modes, self._tonics)
        templates = builder.build_templates()
        self._matcher = SimilarityMatcher(logger, templates)

        penalty_scaled = modulation_penalty * 2.0 / self._modes['Ionian (Major)'].sum()
        self._decoder = StateSequenceDecoder(logger, switching_penalty=penalty_scaled)
        self._merger = RunLengthMerger(logger)

        self._note_extractor = NoteExtractor(logger, subdivisions_per_beat=2, hop_length_samples=512)
        self._segment_profiler = SegmentProfiler(logger)

        self._logger.info("Initialized KeyProgressionAnalyzer.", separator=self._separator)

    def analyze(
            self,
            file_path: Path,
            time_signature: Tuple[int,int] = (4,4),
            segment_beat_level: int = 3
    ) -> List:
        if not file_path.is_file():
            self._logger.error(f"File not found: {file_path}", separator=self._separator)
            return []
        self._logger.info(f"Starting analysis on file: {file_path}", separator=self._separator)

        notes, profiled = self._note_extractor.extract(
            file_path, time_signature, segment_beat_level, visualize=False
        )
        segments = self._segment_profiler.profile_segments(profiled, notes)

        extractor = FeatureVectorExtractor(self._logger, LOFFeatureTransformer())
        vectors, intervals = extractor.extract(segments)

        result = self._matcher.match(vectors)
        path = self._decoder.decode(result.matrix)
        runs = self._merger.merge(intervals, path.tolist(), result.labels)

        self._logger.info("Analysis complete.", separator=self._separator)
        return runs
