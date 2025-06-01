from typing import Tuple, List, Dict

import numpy as np

from track_analysis.components.md_common_python.py_common.algorithms.dynamic_programming import StateSequenceDecoder
from track_analysis.components.md_common_python.py_common.algorithms.sequence import RunLengthMerger
from track_analysis.components.md_common_python.py_common.algorithms.sequence.run_length_merger import StateRun
from track_analysis.components.md_common_python.py_common.algorithms.similarity import SimilarityMatcher
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.key_extraction.core.config.key_progression_config import \
    KeyProgressionConfig
from track_analysis.components.track_analysis.features.key_extraction.core.penalty.penalty_matrix_builder import \
    PenaltyMatrixBuilder
from track_analysis.components.track_analysis.features.key_extraction.core.templates.key_template_builder import \
    KeyTemplateBuilder
from track_analysis.components.track_analysis.features.key_extraction.feature.lof.lof_feature_transformer import \
    LOFFeatureTransformer
from track_analysis.components.track_analysis.features.key_extraction.feature.vector.feature_vector_extractor import \
    FeatureVectorExtractor
from track_analysis.components.track_analysis.features.key_extraction.preprocessing.note_extraction.note_extractor import \
    NoteExtractor
from track_analysis.components.track_analysis.features.key_extraction.preprocessing.profiling.segment_profiler import \
    SegmentProfiler
from track_analysis.components.track_analysis.features.key_extraction.preprocessing.segmentation.audio_segmenter import \
    AudioSegmenter


class LocalKeyEstimator:
    """
    Performs everything needed to get from raw audio + beats to:
      1) a list of StateRun (local key runs),
      2) the underlying intervals,
      3) and the feature‐vector matrix (used later for global key).

    Single responsibility: estimate local key progression—and nothing more.
    """
    def __init__(self, logger: HoornLogger, config: KeyProgressionConfig):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._config = config

        # Build pitch‐class templates for local key decoding:
        template_builder = KeyTemplateBuilder(logger, template_mode=config.template_mode)
        self._local_templates = template_builder.build_templates()

        # A SimilarityMatcher for local decoding:
        self._local_matcher = SimilarityMatcher(logger, self._local_templates)

        # Penalty matrix + DP decoder + run‐length merger:
        penalty_matrix = PenaltyMatrixBuilder(
            logger,
            list(self._local_templates.keys()),
            base_tonic_penalty=config.tone_modulation_penalty,
            base_mode_penalty=config.mode_modulation_penalty,
        ).build()
        self._decoder = StateSequenceDecoder(logger, penalty_matrix=penalty_matrix)
        self._merger = RunLengthMerger(logger)

        # Pre‐built subcomponents for the local pipeline:
        self._note_extractor = NoteExtractor(logger, hop_length_samples=config.hop_length)
        self._audio_segmenter = AudioSegmenter(
            logger,
            config.cache_dir,
            hop_length_samples=config.hop_length,
            subdivisions_per_beat=config.subdivisions_per_beat,
        )
        self._segment_profiler = SegmentProfiler(logger)

        # The feature extractor (wraps LOF transformations):
        self._feature_extractor = FeatureVectorExtractor(logger, LOFFeatureTransformer())

        self._logger.info("Initialized LocalKeyEstimator.", separator=self._separator)

    def get_local_templates(self) -> Dict[str, np.ndarray]:
        return self._local_templates

    def analyze(
            self,
            audio_samples: np.ndarray,
            sample_rate: int,
            track_tempo: float,
            beat_frames: np.ndarray,
            beat_times: np.ndarray,
    ) -> Tuple[List[StateRun], List[Tuple[float, float]], List[np.ndarray]]:
        """
        1) Extract notes from raw audio & tempo,
        2) Segment audio into chunks at beat‐level = config.segment_beat_level,
        3) Profile each segment to collect note/chroma statistics,
        4) Extract feature vectors,
        5) Match each vector against local templates (SimilarityMatcher),
        6) Decode via Viterbi (StateSequenceDecoder) + merge runs (RunLengthMerger).
        """
        self._logger.info("Starting local key analysis.", separator=self._separator)

        # --- 1. Note extraction ---
        notes = self._note_extractor.extract(
            audio_samples, sample_rate, track_tempo, visualize=self._config.visualize
        )

        # --- 2. Audio segmentation (at beat granularity) ---
        segments = self._audio_segmenter.get_segments(
            beat_frames, beat_times, audio_samples, sample_rate,
            min_segment_level=self._config.segment_beat_level
        )

        # --- 3. Profile each segment (collect note/chroma info) ---
        profiled_segments = self._segment_profiler.profile_segments(segments, notes)

        # --- 4. Extract feature vectors + keep intervals for merging ---
        feature_matrix, intervals = self._feature_extractor.extract_segments(profiled_segments)

        # --- 5. Local similarity matching & decoding ---
        local_match_result = self._local_matcher.match(feature_matrix)
        local_path = self._decoder.decode(local_match_result.matrix)

        # --- 6. Merge consecutive identical states into runs ---
        local_runs = self._merger.merge(
            intervals,
            local_path.tolist(),
            local_match_result.labels
        )

        self._logger.info("Finished local key analysis.", separator=self._separator)
        return local_runs, intervals, feature_matrix
