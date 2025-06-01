from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from track_analysis.components.md_common_python.py_common.algorithms.dynamic_programming import StateSequenceDecoder
from track_analysis.components.md_common_python.py_common.algorithms.sequence import RunLengthMerger
from track_analysis.components.md_common_python.py_common.algorithms.sequence.run_length_merger import StateRun
from track_analysis.components.md_common_python.py_common.algorithms.similarity import SimilarityMatcher
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.key_extraction.core.definitions.definition_templates import \
    TemplateMode
from track_analysis.components.track_analysis.features.key_extraction.core.penalty.penalty_matrix_builder import \
    PenaltyMatrixBuilder
from track_analysis.components.track_analysis.features.key_extraction.core.templates.key_template_builder import \
    KeyTemplateBuilder
from track_analysis.components.track_analysis.features.key_extraction.feature.vector.feature_vector_extractor import \
    FeatureVectorExtractor
from track_analysis.components.track_analysis.features.key_extraction.feature.transforming.lof_feature_transformer import \
    LOFFeatureTransformer
from track_analysis.components.track_analysis.features.key_extraction.preprocessing.note_extraction.note_extractor import \
    NoteExtractor
from track_analysis.components.track_analysis.features.key_extraction.preprocessing.profiling.segment_profiler import \
    SegmentProfiler


class KeyProgressionAnalyzer:
    """
    Orchestrates key progression analysis: extraction, matching, decoding, merging,
    and infers a global key for the entire piece.
    """
    def __init__(
            self,
            logger: HoornLogger,
            tone_modulation_penalty: float = 6.0,
            mode_modulation_penalty: Optional[float] = None,
            visualize: bool = False,
            template_mode: TemplateMode = TemplateMode.KS_T_REVISED
    ):
        self._visualize = visualize

        self._logger = logger
        self._separator = self.__class__.__name__

        # Build pitch-class templates for local key decoding
        builder = KeyTemplateBuilder(logger, template_mode=template_mode)
        self._local_templates = builder.build_templates()
        self._local_matcher = SimilarityMatcher(logger, self._local_templates)
        self._global_matcher = SimilarityMatcher(
            logger=self._logger,
            templates=self._local_templates,
            label_order=list(self._local_templates.keys()),
            verbose=False
        )

        if mode_modulation_penalty is None:
            mode_modulation_penalty_scale = 2.5 / 6
            mode_modulation_penalty = mode_modulation_penalty_scale * tone_modulation_penalty

        penalty_matrix = PenaltyMatrixBuilder(
            logger,
            list(self._local_templates.keys()),
            base_tonic_penalty=tone_modulation_penalty,
            base_mode_penalty=mode_modulation_penalty
        ).build()

        self._penalty_matrix = penalty_matrix
        self._decoder = StateSequenceDecoder(logger, penalty_matrix=penalty_matrix)
        self._merger = RunLengthMerger(logger)

        self._note_extractor = NoteExtractor(logger, subdivisions_per_beat=2, hop_length_samples=512)
        self._segment_profiler = SegmentProfiler(logger)

        self._logger.info("Initialized KeyProgressionAnalyzer.", separator=self._separator)

    def analyze(
            self,
            file_path: Path,
            segment_beat_level: int = 3
    ) -> Tuple[List[StateRun], Optional[str]]:
        """
        Returns a tuple:
          - List of StateRun for local key progressions
          - global key label (str) or None if analysis failed
        """
        if not file_path.is_file():
            self._logger.error(f"File not found: {file_path}", separator=self._separator)
            return [], None
        self._logger.info(f"Starting analysis on file: {file_path}", separator=self._separator)

        # --- Local key pass ---
        notes, profiled = self._note_extractor.extract(
            file_path, segment_beat_level, visualize=self._visualize
        )
        segments = self._segment_profiler.profile_segments(profiled, notes)

        extractor = FeatureVectorExtractor(self._logger, LOFFeatureTransformer())
        vectors, intervals = extractor.extract(segments)

        local_match = self._local_matcher.match(vectors)
        local_path = self._decoder.decode(local_match.matrix)
        local_runs = self._merger.merge(intervals, local_path.tolist(), local_match.labels)
        self._logger.info("Local key analysis complete.", separator=self._separator)

        # --- Global key second pass ---
        # 1. Weight each segment‐vector by its duration to form one global chroma:
        durations = [end - start for (start, end) in intervals]

        global_chroma = np.zeros(12, dtype=float)
        for i, vec in enumerate(vectors):
            global_chroma += durations[i] * vec

        # Normalize so it has unit L1 norm (or L2 if your templates expect that)
        global_chroma /= np.linalg.norm(global_chroma, ord=1)
        score_result = self._global_matcher.match([global_chroma])

        scores = score_result.matrix[0]
        labels = score_result.labels

        # 3. Pick the label with the highest score:
        best_idx = int(np.argmax(scores))
        global_key = labels[best_idx]

        self._logger.info("Analysis complete.", separator=self._separator)
        return local_runs, global_key
