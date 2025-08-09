from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from track_analysis.components.md_common_python.py_common.algorithms.similarity import SimilarityMatcher
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.utils.key_to_camelot import \
    convert_label_to_camelot
from track_analysis.components.track_analysis.shared.caching.hdf5_memory import TimedCacheResult
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.timed_cache(identifier_arg="file_path", ignore=["feature_matrix"])
def _compute_global_chroma(
        *,
        file_path: Path,
        feature_matrix: np.ndarray,
) -> TimedCacheResult[np.ndarray]:
    global_chroma = np.zeros(12, dtype=float)
    for i, vec in enumerate(feature_matrix):
        global_chroma += vec

    # 3. Normalize to L1‐norm
    norm = np.linalg.norm(global_chroma, ord=1)
    if norm > 0.0:
        global_chroma /= norm

    return global_chroma


class GlobalKeyProvider(AudioDataFeatureProvider):
    def __init__(self, logger: HoornLogger, similarity_matcher: SimilarityMatcher):
        super().__init__()
        self._logger = logger
        self._separator = self.__class__.__name__

        self._global_matcher = similarity_matcher
        self._logger.trace("Initialized GlobalKeyEstimator.", separator=self._separator)

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.TRACK_FEATURE_VECTOR, AudioDataFeature.AUDIO_PATH]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return AudioDataFeature.PRINCIPAL_KEY

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            audio_path: Path = data[AudioDataFeature.AUDIO_PATH]
            feature_matrix: np.ndarray = data[AudioDataFeature.TRACK_FEATURE_VECTOR]

        global_chroma_results = _compute_global_chroma(file_path=audio_path, feature_matrix=feature_matrix)
        self._add_timed_cache_times(global_chroma_results)

        with self._measure_processing():
            score_result = self._global_matcher.match([global_chroma_results.value])
            scores = score_result.matrix[0]
            labels = score_result.labels
            best_idx = int(np.argmax(scores))
            best_label = labels[best_idx]

            self._logger.info(f"Global key estimated: {best_label}", separator=self._separator)
            return {
                AudioDataFeature.PRINCIPAL_KEY: convert_label_to_camelot(best_label),
            }

