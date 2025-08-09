from typing import List, Tuple

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.timing.configuration.timing_analysis_configuration import \
    TimingAnalysisConfiguration
from track_analysis.components.track_analysis.library.timing.model.processed_feature import ProcessedFeature


class SuggestionEngine:
    def __init__(self, logger: HoornLogger, configuration: TimingAnalysisConfiguration):
        self._logger = logger
        self._separator: str = self.__class__.__name__
        self._configuration = configuration

    def generate_suggestions(self, features: List[ProcessedFeature], total_feature_time: float) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Analyzes features to provide optimization and investigation suggestions.
        """
        if not total_feature_time:
            return [], []

        # 1. Gather candidates for each category
        wait_candidates = self._get_wait_candidates(features, total_feature_time)
        optimize_candidates = self._get_optimize_candidates(features, total_feature_time)

        # 2. Format the gathered candidates into a user-friendly list
        return wait_candidates, optimize_candidates

    def _is_feature_significant(self, feature: ProcessedFeature, total_feature_time: float) -> bool:
        """Checks if a feature's performance is significant enough to warrant a suggestion."""
        impact_percent = (feature.total_time / total_feature_time) * 100

        is_impactful = impact_percent >= self._configuration.minimum_impact_percentage_threshold
        is_long_enough = feature.total_time >= self._configuration.minimum_total_time_spent_threshold

        return is_impactful and is_long_enough

    def _get_wait_candidates(self, features: List[ProcessedFeature], total_feature_time: float) -> List[Tuple[str, str]]:
        """Identifies features that are potentially blocked (I/O bound)."""
        candidates = []
        for f in features:
            is_significant = self._is_feature_significant(f, total_feature_time)
            is_waiting_more_than_processing = f.wait_time > f.process_time

            wait_ratio = self._get_time_ratio(f.wait_time, f.total_time)

            high_enough_wait_ratio = wait_ratio > self._configuration.minimum_spent_ratio

            if is_significant and is_waiting_more_than_processing and high_enough_wait_ratio:
                reason = (
                    f"Spent {wait_ratio:.0%} of its time waiting "
                    f"({f.wait_time:.2f}s of {f.total_time:.2f}s total)."
                )
                candidates.append((f.name, reason))

        return candidates

    def _get_optimize_candidates(self, features: List[ProcessedFeature], total_feature_time: float) -> List[Tuple[str, str]]:
        """Identifies features that are potentially CPU-bound."""
        candidates = []
        for f in features:
            is_feature_significant = self._is_feature_significant(f, total_feature_time)
            is_processing_more_than_waiting = f.process_time > f.wait_time

            process_ratio = self._get_time_ratio(f.process_time, f.total_time)

            high_enough_process_ratio = process_ratio > self._configuration.minimum_spent_ratio

            if is_feature_significant and is_processing_more_than_waiting and high_enough_process_ratio:
                reason = (
                    f"Spent {process_ratio:.0%} of its time processing "
                    f"({f.process_time:.2f}s of {f.total_time:.2f}s total)."
                )
                candidates.append((f.name, reason))

        return candidates

    @staticmethod
    def _get_time_ratio(time_to_check: float, total_time: float):
        return time_to_check / total_time if total_time > 0 else 0
