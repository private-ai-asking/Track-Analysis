from typing import List, Tuple

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.timing.configuration.timing_analysis_configuration import \
    TimingAnalysisConfiguration
from track_analysis.components.track_analysis.library.timing.model.processed_feature import ProcessedFeature


class ReportFormatter:
    def __init__(self, logger: HoornLogger, configuration: TimingAnalysisConfiguration):
        self._logger = logger
        self._separator: str = self.__class__.__name__
        self._configuration: TimingAnalysisConfiguration = configuration

    def log_report(
            self, batch_size: int, total_time: float, own_wait: float, own_proc: float,
            feature_time: float, features: List[ProcessedFeature],
            wait_candidates: List[Tuple[str, str]], optimize_candidates: List[Tuple[str, str]]
    ):
        """Formats and logs the final analysis report, including suggestions."""
        own_total = own_wait + own_proc
        unaccounted_time = total_time - own_total - feature_time
        lines = ["\n--- Timing Analysis Report ---"]

        # General Summary
        plural: bool = batch_size > 1
        avg_run_time = total_time / batch_size if batch_size > 0 else 0
        lines.append(f"Analyzed {batch_size} {self._configuration.timing_data_name_plural if plural else self._configuration.timing_data_name}. Total time: {total_time:.3f}s. Average per run: {avg_run_time:.3f}s.")

        # Overall Time Distribution
        if total_time > 0:
            lines.append("\nOverall Time Distribution:")
            lines.append(f"  - Main Logic:     {own_total:8.3f}s ({(own_total / total_time) * 100:5.1f}%) -> [Wait: {own_wait:.3f}s, Process: {own_proc:.3f}s]")
            lines.append(f"  - Sub-features:   {feature_time:8.3f}s ({(feature_time / total_time) * 100:5.1f}%)")
            if unaccounted_time > 0.001:
                lines.append(f"  - Unaccounted:    {unaccounted_time:8.3f}s ({(unaccounted_time / total_time) * 100:5.1f}%)")

        # Feature Table
        lines.append("\n--- Feature Dissection (sorted by Total Time) ---")
        lines.extend(self._create_feature_table(features, feature_time))

        # Suggestions Section
        suggestions = self._format_suggestions(wait_candidates, optimize_candidates)
        if suggestions:
            lines.append("")  # Add vertical space
            lines.extend(suggestions)

        lines.append("\n--- End of Report ---")
        self._logger.info("\n".join(lines), separator=self._separator)

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

    def _format_suggestions(
            self,
            wait_candidates: List[Tuple[str, str]],
            optimize_candidates: List[Tuple[str, str]]
    ) -> List[str]:
        """Formats the final list of suggestions from the identified candidates."""
        if not wait_candidates and not optimize_candidates:
            return []

        suggestions = ["--- 💡 Suggestions ---"]

        suggestions.extend(self._format_suggestion_section(
            candidates=wait_candidates,
            header="\n[Investigate High Wait Times]",
            description="These features are blocked. Investigate their code to find the root cause, which could be I/O, a slow cache, or resource contention. Consider using `asyncio` or threading to run this task concurrently (and investigate the source of the waiting)."
        ))

        suggestions.extend(self._format_suggestion_section(
            candidates=optimize_candidates,
            header="\n[Consider Optimizing Code]",
            description="These features are CPU-bound. Consider optimizing the algorithm or using more efficient libraries."
        ))

        return suggestions

    @staticmethod
    def _format_suggestion_section(
            candidates: List[Tuple[str, str]],
            header: str,
            description: str
    ) -> List[str]:
        """Formats a list of candidate tuples into a complete suggestion section."""
        if not candidates:
            return []

        section = [header, description]
        for name, reason in candidates:
            section.append(f"  - '{name}': {reason}")
        return section
