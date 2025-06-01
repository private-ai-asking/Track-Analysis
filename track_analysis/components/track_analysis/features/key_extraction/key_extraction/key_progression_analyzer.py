from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from track_analysis.components.md_common_python.py_common.algorithms.dynamic_programming import StateSequenceDecoder
from track_analysis.components.md_common_python.py_common.algorithms.sequence import RunLengthMerger
from track_analysis.components.md_common_python.py_common.algorithms.sequence.run_length_merger import StateRun
from track_analysis.components.md_common_python.py_common.algorithms.similarity import SimilarityMatcher
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.key_extraction.key_extraction.feature_vector_extractor import \
    FeatureVectorExtractor
from track_analysis.components.track_analysis.features.key_extraction.key_extraction.key_template_builder import \
    KeyTemplateBuilder
from track_analysis.components.track_analysis.features.key_extraction.key_extraction.lof_feature_transformer import \
    LOFFeatureTransformer
from track_analysis.components.track_analysis.features.key_extraction.key_extraction.penalty_matrix_builder import \
    PenaltyMatrixBuilder
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.note_extractor import \
    NoteExtractor
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.segment_profiler import \
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
            modes: Dict[str, np.ndarray] = None,
            tonics: List[str] = None,
            visualize: bool = False,
    ):
        self._visualize = visualize

        self._logger = logger
        self._separator = self.__class__.__name__
        # default music modes and tonics
        self._modes = modes or {
            # K&S - Temperley Revised
            # 'Ionian (Major)': np.array([5.0,2.0,3.5,2.0,4.5,4.0,2.0,4.5,2.0,3.5,1.5,4.0]),
            # 'Aeolian (Minor)': np.array([5.0,2.0,3.5,4.5,2.0,4.0,2.0,4.5,3.5,2.0,1.5,4.0]),

            # Bellman-Budge
            'Ionian (Major)': np.array([16.80, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 20.28, 1.80, 8.04, 0.62, 10.57]),
            'Aeolian (Minor)': np.array([18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 21.07, 7.49, 1.53, 0.92, 10.21])
        }
        self._tonics_order_of_fifths = tonics or ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']

        # C, C♯, D, D♯, E, F, F♯, G, G♯, A, A♯, B
        # 0, 1,  2, 3,  4, 5, 6,  7, 8,  9, 10, 11

        # C, D#, F, G, A#

        # Build pitch-class templates for local key decoding
        builder = KeyTemplateBuilder(logger, self._modes, self._tonics_order_of_fifths)
        self._local_templates = builder.build_templates()
        self._local_matcher = SimilarityMatcher(logger, self._local_templates)
        self._global_matcher = SimilarityMatcher(
            logger=self._logger,
            templates=self._local_templates,
            label_order=list(self._local_templates.keys()),
            verbose=False
        )

        mode_penalties: Dict[str, Dict[str, float]] = {
            "Ionian (Major)": {
                "Ionian (Major)": 0.0,
                "Aeolian (Minor)": 3.0,
                # "Dorian (Minor)": 2.0,
            },
            "Aeolian (Minor)": {
                "Aeolian (Minor)": 0.0,
                "Ionian (Major)": 3.0,
                # "Dorian (Minor)": 1.0,
            },
            # "Dorian (Minor)": {
            #     "Dorian (Minor)": 0.0,
            #     "Ionian (Major)": 2.0,
            #     "Aeolian (Minor)": 1.0,
            # }
        }

        if mode_modulation_penalty is None:
            mode_modulation_penalty_scale = 2.5 / 6
            mode_modulation_penalty = mode_modulation_penalty_scale * tone_modulation_penalty

        tone_penalty_scaled = tone_modulation_penalty * 2.0 / self._modes['Ionian (Major)'].sum()
        mode_penalty_scaled = mode_modulation_penalty * 2.0 / self._modes['Ionian (Major)'].sum()
        penalty_matrix = PenaltyMatrixBuilder(
            logger,
            list(self._local_templates.keys()),
            self._tonics_order_of_fifths,
            base_tonic_penalty=tone_penalty_scaled,
            base_mode_penalty=mode_penalty_scaled,
            mode_penalty_matrix=mode_penalties,
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
