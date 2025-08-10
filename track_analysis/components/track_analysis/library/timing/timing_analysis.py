import collections
from typing import Iterable, List, Dict, Optional

import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger, LogType
from track_analysis.components.md_common_python.py_common.time_handling import TimeUtils
from track_analysis.components.track_analysis.library.timing.configuration.timing_analysis_configuration import \
    TimingAnalysisConfiguration
from track_analysis.components.track_analysis.library.timing.model.analysis_report_data import \
    AnalysisReportData, ParallelismStats
from track_analysis.components.track_analysis.library.timing.model.processed_feature import ProcessedFeature
from track_analysis.components.track_analysis.library.timing.model.timing_data import TimingData
from track_analysis.components.track_analysis.library.timing.report_formatter import ReportFormatter
from track_analysis.components.track_analysis.library.timing.suggestion_engine import SuggestionEngine


class TimingAnalyzer:
    """
    Analyzes and logs execution timing, feature performance, and parallelism efficiency.
    """
    def __init__(self, logger: HoornLogger, configuration: TimingAnalysisConfiguration = TimingAnalysisConfiguration()):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._suggestion_engine = SuggestionEngine(self._logger, configuration)
        self._report_formatter = ReportFormatter(self._logger, configuration, TimeUtils())

    def analyze_time(self,
                     timing_data_batch: Iterable[TimingData],
                     wall_clock_time: float = 0.0,
                     worker_count: int = 1,
                     log_mode: LogType = LogType.INFO):
        """Analyzes a batch of timing data and logs a comprehensive performance report.
        You can also provide a batch of 1 to only analyze a single entry.
        """
        timing_data_list = list(timing_data_batch)
        if not timing_data_list:
            self._logger.warning("analyze_time_batch called with an empty iterable.", separator=self._separator)
            return

        # 1. Aggregate core timing data
        batch_size = len(timing_data_list)
        total_sequential_time = sum(d.total_time_spent for d in timing_data_list)
        total_own_waiting = sum(d.own_waiting_time for d in timing_data_list)
        total_own_processing = sum(d.own_processing_time for d in timing_data_list)

        # 2. Aggregate and process feature-level stats
        feature_stats: Dict[str, ProcessedFeature] = collections.defaultdict(ProcessedFeature)
        for data in timing_data_list:
            for feature in data.feature_dissection:
                pf = feature_stats[feature.associated_feature]
                pf.name = feature.associated_feature
                pf.wait_time += feature.time_spent_waiting
                pf.process_time += feature.time_spent_processing
                pf.call_count += 1
                pf.all_timings_ms.append((feature.time_spent_waiting + feature.time_spent_processing) * 1000)

        processed_features: List[ProcessedFeature] = []
        for name, pf in feature_stats.items():
            pf.finalize_stats()
            processed_features.append(pf)
        processed_features.sort(key=lambda x: x.total_time, reverse=True)
        total_feature_time = sum(f.total_time for f in processed_features)

        # 3. Calculate parallelism stats if applicable
        parallelism_stats = self._calculate_parallelism_stats(
            wall_clock_time, total_sequential_time, batch_size, worker_count
        )

        # 4. Generate suggestions
        suggestions = self._suggestion_engine.generate_suggestions(processed_features, total_feature_time)

        # 5. Assemble the final report data object
        report_data = AnalysisReportData(
            batch_size=batch_size,
            total_sequential_time=total_sequential_time,
            total_own_waiting=total_own_waiting,
            total_own_processing=total_own_processing,
            total_feature_time=total_feature_time,
            processed_features=processed_features,
            suggestions=suggestions,
            parallelism_stats=parallelism_stats
        )

        # 6. Pass the single data object to the formatter
        self._report_formatter.log_report(report_data, log_mode)

    @staticmethod
    def _calculate_parallelism_stats(wall_clock_time: float,
                                     sequential_time: float,
                                     task_count: int,
                                     worker_count: int) -> Optional[ParallelismStats]:
        """Calculates parallelism efficiency metrics if the run was parallel."""
        if wall_clock_time <= 0 or worker_count <= 1:
            return None

        individual_task_times = [sequential_time / task_count] * task_count

        speedup_factor = sequential_time / wall_clock_time
        utilization = (speedup_factor / worker_count) * 100

        return ParallelismStats(
            wall_clock_time=wall_clock_time,
            sequential_time=sequential_time,
            task_count=task_count,
            worker_count=worker_count,
            speedup_factor=speedup_factor,
            avg_throughput=task_count / wall_clock_time,
            worker_utilization_percent=utilization,
            total_overhead_time=(wall_clock_time * worker_count) - sequential_time,
            min_task_time=np.min(individual_task_times) if individual_task_times else 0,
            max_task_time=np.max(individual_task_times) if individual_task_times else 0,
            mean_task_time=np.mean(individual_task_times) if individual_task_times else 0,
            median_task_time=np.median(individual_task_times) if individual_task_times else 0,
            stdev_task_time=np.std(individual_task_times) if individual_task_times else 0,
        )
