from typing import List, Tuple

from track_analysis.components.md_common_python.py_common.logging import HoornLogger, LogType
from track_analysis.components.md_common_python.py_common.time_handling import TimeUtils
from track_analysis.components.track_analysis.library.timing.configuration.timing_analysis_configuration import \
    TimingAnalysisConfiguration
from track_analysis.components.track_analysis.library.timing.model.analysis_report_data import AnalysisReportData
from track_analysis.components.track_analysis.library.timing.model.processed_feature import ProcessedFeature
from track_analysis.components.track_analysis.library.timing.model.suggestion_categories import SuggestionCategories


class ReportFormatter:
    def __init__(self, logger: HoornLogger, configuration: TimingAnalysisConfiguration, time_utils: TimeUtils):
        self._logger = logger
        self._separator: str = self.__class__.__name__
        self._configuration: TimingAnalysisConfiguration = configuration
        self._time_utils: TimeUtils = time_utils

    def log_report(self, data: AnalysisReportData, log_mode: LogType):
        """Formats and logs the final analysis report from a single data object."""
        lines = ["\n--- Timing Analysis Report ---"]

        # General Summary
        plural = data.batch_size > 1
        avg_run_time = data.total_sequential_time / data.batch_size if data.batch_size > 0 else 0
        lines.append(f"Analyzed {data.batch_size} {self._configuration.timing_data_name_plural if plural else self._configuration.timing_data_name}. "
                     f"Avg task time: {self._time_utils.format_time(avg_run_time)}.")

        # Parallelism Analysis Section
        if data.parallelism_stats:
            p_stats = data.parallelism_stats
            lines.append("\n--- Parallelism Analysis ---")
            lines.append(f"🚀 Speedup: {p_stats.speedup_factor:.2f}x (1x = baseline) | "
                         f"Throughput: {p_stats.avg_throughput:.2f} tasks/sec")
            lines.append(f"🔋 Worker Utilization: {p_stats.worker_utilization_percent:.1f}% | "
                         f"Overhead: {self._time_utils.format_time(p_stats.total_overhead_time)}")
            lines.append(f"   - Sequential Time: {self._time_utils.format_time(p_stats.sequential_time)} (on 1 worker)")
            lines.append(f"   - Parallel Time:   {self._time_utils.format_time(p_stats.wall_clock_time)} (on {p_stats.worker_count} workers)")

        # Overall Time Distribution
        own_total = data.total_own_waiting + data.total_own_processing
        unaccounted_time = data.total_sequential_time - own_total - data.total_feature_time
        if data.total_sequential_time > 0:
            lines.append("\n--- Overall Time Distribution (Based on Sequential Time) ---")
            lines.append(f"  - Main Logic:     {own_total:8.3f}s ({(own_total / data.total_sequential_time) * 100:5.1f}%) -> "
                         f"[Wait: {data.total_own_waiting:.3f}s, Process: {data.total_own_processing:.3f}s]")
            lines.append(f"  - Sub-features:   {data.total_feature_time:8.3f}s ({(data.total_feature_time / data.total_sequential_time) * 100:5.1f}%)")
            if unaccounted_time > 0.001:
                lines.append(f"  - Unaccounted:    {unaccounted_time:8.3f}s ({(unaccounted_time / data.total_sequential_time) * 100:5.1f}%)")

        # Feature Table
        lines.append("\n--- Feature Dissection (sorted by Total Time) ---")
        lines.extend(self._create_feature_table(data.processed_features, data.total_feature_time))

        # Suggestions Section
        if any(vars(data.suggestions).values()):
            lines.append("")
            lines.extend(self._format_suggestions(data.suggestions))

        lines.append("\n--- End of Report ---")
        self._logger.log_raw(log_mode, "\n".join(lines), separator=self._separator)

    @staticmethod
    def _create_feature_table(features: List[ProcessedFeature], total_feature_time: float) -> List[str]:
        """Formats the detailed feature data into a text-based table."""
        if not features:
            return ["No sub-feature timing data provided."]

        max_name_len = max((len(f.name) for f in features), default=0)
        max_name_len = max(max_name_len, len("Feature"))

        header = (
            f"{'Feature':<{max_name_len}} | {'Total (s)':>10} | {'% of Feat.':>10} | {'Avg (ms)':>10} | "
            f"{'StDev (ms)':>10} | {'Max (ms)':>10} | {'Calls':>8}"
        )
        table = [header, '-' * len(header)]

        for f in features:
            perc_of_total = (f.total_time / total_feature_time * 100) if total_feature_time > 0 else 0
            row = (
                f"{f.name:<{max_name_len}} | {f.total_time:>10.3f} | {perc_of_total:>9.1f}% | "
                f"{f.avg_time_ms:>10.2f} | {f.stdev_ms:>10.2f} | {f.max_time_ms:>10.2f} | {f.call_count:>8}"
            )
            table.append(row)

        return table

    def _format_suggestions(self, suggestions: SuggestionCategories) -> List[str]:
        """Formats the final list of suggestions from all identified categories."""
        all_candidates = (
                suggestions.wait_candidates + suggestions.optimize_candidates +
                suggestions.variance_candidates + suggestions.caching_candidates
        )
        if not all_candidates:
            return []

        suggestion_lines = ["--- 💡 Suggestions ---"]

        suggestion_lines.extend(self._format_suggestion_section(
            candidates=suggestions.wait_candidates,
            header="\n[Investigate High Wait Times]",
            description="These features are blocked, likely by I/O. Consider using `asyncio` or threading to perform work concurrently."
        ))

        suggestion_lines.extend(self._format_suggestion_section(
            candidates=suggestions.optimize_candidates,
            header="\n[Consider Optimizing Code]",
            description="These features are CPU-bound. Profile them to find hot spots and consider optimizing the algorithm or using more efficient libraries."
        ))

        suggestion_lines.extend(self._format_suggestion_section(
            candidates=suggestions.variance_candidates,
            header="\n[Investigate Performance Variance]",
            description="These features have inconsistent performance. Investigate if their runtime correlates with input data characteristics."
        ))

        suggestion_lines.extend(self._format_suggestion_section(
            candidates=suggestions.caching_candidates,
            header="\n[Consider Caching Results]",
            description="These features have a high CPU cost. If their output is deterministic, consider caching results to avoid re-computation on subsequent runs."
        ))

        return suggestion_lines

    @staticmethod
    def _format_suggestion_section(candidates: List[Tuple[str, str]], header: str, description: str) -> List[str]:
        """Formats a list of candidate tuples into a complete suggestion section."""
        if not candidates:
            return []

        section = [header, description]
        for name, reason in candidates:
            section.append(f"  - '{name}': {reason}")
        return section
