from typing import List, Tuple, Callable

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
        wait_candidates = self._get_candidates(
            features,
            total_feature_time,
            condition=lambda f: f.wait_time > f.process_time,
            time_metric=lambda f: f.wait_time,
            reason_template="Spent {:.0%} of its time waiting ({:.2f}s of {:.2f}s total)."
        )

        optimize_candidates = self._get_candidates(
            features,
            total_feature_time,
            condition=lambda f: f.process_time > f.wait_time,
            time_metric=lambda f: f.process_time,
            reason_template="Spent {:.0%} of its time processing ({:.2f}s of {:.2f}s total). "
        )

        # 2. Format the gathered candidates into a user-friendly list
        return wait_candidates, optimize_candidates

    def _is_feature_significant(self, feature: ProcessedFeature, total_feature_time: float) -> bool:
        """Checks if a feature's performance is significant enough to warrant a suggestion."""
        impact_percent = (feature.total_time / total_feature_time) * 100

        is_impactful = impact_percent >= self._configuration.minimum_impact_percentage_threshold
        is_long_enough = feature.total_time >= self._configuration.minimum_total_time_spent_threshold

        return is_impactful and is_long_enough

    def _get_candidates(
            self, features: List[ProcessedFeature], total_feature_time: float,
            condition: Callable[[ProcessedFeature], bool],
            time_metric: Callable[[ProcessedFeature], float],
            reason_template: str
    ) -> List[Tuple[str, str]]:
        """Generic method to identify suggestion candidates."""
        candidates = []
        for f in features:
            if not self._is_feature_significant(f, total_feature_time) or not condition(f):
                continue

            ratio = self._get_time_ratio(time_metric(f), f.total_time)
            if ratio > self._configuration.minimum_spent_ratio:
                reason = reason_template.format(ratio, time_metric(f), f.total_time)
                candidates.append((f.name, reason))
        return candidates

    @staticmethod
    def _get_time_ratio(time_to_check: float, total_time: float):
        return time_to_check / total_time if total_time > 0 else 0
