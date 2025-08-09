import dataclasses
from typing import List, Dict, Any, Tuple

import numpy as np

from track_analysis.components.md_common_python.py_common.algorithms.dynamic_programming import StateSequenceDecoder
from track_analysis.components.md_common_python.py_common.algorithms.sequence import RunLengthMerger
from track_analysis.components.md_common_python.py_common.algorithms.sequence.run_length_merger import StateRun
from track_analysis.components.md_common_python.py_common.algorithms.similarity import SimilarityMatcher
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.core.config.key_progression_config import \
    KeyProgressionConfig
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.penalties.penalty_matrix_builder import \
    PenaltyMatrixBuilder
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.preprocessing.profiling.segment_profiler import \
    ProfiledSegment
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.utils.key_to_camelot import \
    convert_label_to_camelot


class LocalKeyProvider(AudioDataFeatureProvider):
    def __init__(self, logger: HoornLogger, config: KeyProgressionConfig, templates: Dict[str, np.ndarray],
                 similarity_matcher: SimilarityMatcher):
        super().__init__()
        self._logger = logger
        self._separator = self.__class__.__name__
        self._config = config
        self._template_matcher = similarity_matcher

        penalty_matrix = PenaltyMatrixBuilder(
            logger,
            list(templates.keys()),
            base_tonic_penalty=config.tone_modulation_penalty,
            base_mode_penalty=config.mode_modulation_penalty,
        ).build()

        self._decoder = StateSequenceDecoder(logger, penalty_matrix=penalty_matrix)
        self._merger = RunLengthMerger(logger)

        self._logger.trace("Initialized LocalKeyEstimator.", separator=self._separator)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.TRACK_FEATURE_VECTOR, AudioDataFeature.TRACK_SEGMENTS_PROFILED]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return [AudioDataFeature.KEY_PROGRESSION, AudioDataFeature.START_KEY, AudioDataFeature.END_KEY]

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            feature_matrix = data[AudioDataFeature.TRACK_FEATURE_VECTOR]
            segments: List[ProfiledSegment] = data[AudioDataFeature.TRACK_SEGMENTS_PROFILED]

            intervals: List[Tuple[float, float]] = []

            for segment in segments:
                intervals.append((segment.raw_segment.segment_start_seconds, segment.raw_segment.segment_end_seconds))

            local_match_result = self._template_matcher.match(feature_matrix)
            local_key_progression = self._decoder.decode(local_match_result.matrix)

            local_runs = self._merger.merge(
                intervals,
                local_key_progression.tolist(),
                local_match_result.labels
            )

            formatted_runs: List[StateRun] = [
                dataclasses.replace(run, state_label=convert_label_to_camelot(run.state_label))
                for run in local_runs
            ]
            start_key = formatted_runs[0].state_label
            end_key = formatted_runs[-1].state_label

            self._logger.info("Finished local key analysis.", separator=self._separator)
            return {
                AudioDataFeature.KEY_PROGRESSION: formatted_runs,
                AudioDataFeature.START_KEY: start_key,
                AudioDataFeature.END_KEY: end_key,
            }
