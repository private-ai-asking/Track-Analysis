import collections
from typing import Iterable, List, Dict

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.timing.configuration.timing_analysis_configuration import \
    TimingAnalysisConfiguration
from track_analysis.components.track_analysis.library.timing.model.processed_feature import ProcessedFeature
from track_analysis.components.track_analysis.library.timing.model.timing_data import TimingData
from track_analysis.components.track_analysis.library.timing.report_formatter import ReportFormatter
from track_analysis.components.track_analysis.library.timing.suggestion_engine import SuggestionEngine


class TimingAnalyzer:
    """Helper class to analyze and log execution timing information.

    Provides useful statistics about timing information and highlights features that are
    useful to optimize and/or cache on-disk.
    """

    def __init__(self, logger: HoornLogger, configuration: TimingAnalysisConfiguration = TimingAnalysisConfiguration()):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._configuration = configuration

        self._suggestion_engine: SuggestionEngine = SuggestionEngine(self._logger, self._configuration)
        self._report_formatter: ReportFormatter = ReportFormatter(self._logger)

    def analyze_time(self, timing_data: TimingData):
        """Analyzes a single piece of timing data.

        If you want to analyze multiple pieces of related/grouped timing data, it is recommended
        to use analyze_time_batch instead of calling this multiple times.
        """
        self.analyze_time_batch([timing_data])

    def analyze_time_batch(self, timing_data_batch: Iterable[TimingData]):
        """Analyzes a batch of pieces of timing data and logs a formatted report."""
        timing_data_list = list(timing_data_batch)
        if not timing_data_list:
            self._logger.warning("analyze_time_batch called with an empty iterable.", separator=self._separator)
            return

        batch_size = len(timing_data_list)
        total_time = sum(d.total_time_spent for d in timing_data_list)
        total_own_waiting = sum(d.own_waiting_time for d in timing_data_list)
        total_own_processing = sum(d.own_processing_time for d in timing_data_list)

        # 1. Aggregate all timing data from the batch
        feature_stats: Dict[str, ProcessedFeature] = collections.defaultdict(ProcessedFeature)
        for data in timing_data_list:
            for feature in data.feature_dissection:
                processed_feature = feature_stats[feature.associated_feature]
                processed_feature.name = feature.associated_feature
                processed_feature.wait_time += feature.time_spent_waiting
                processed_feature.process_time += feature.time_spent_processing
                processed_feature.call_count += 1

        # 2. Process aggregated data
        processed_features: List[ProcessedFeature] = []
        for name, processed_feature in feature_stats.items():
            processed_feature.total_time = processed_feature.wait_time + processed_feature.process_time
            processed_feature.avg_time_ms = (processed_feature.total_time / processed_feature.call_count * 1000) if processed_feature.call_count > 0 else 0
            processed_features.append(processed_feature)

        total_feature_time = sum(f.total_time for f in processed_features)
        processed_features.sort(key=lambda x: x.total_time, reverse=True)

        wait_candidates, optimize_candidates = self._suggestion_engine.generate_suggestions(processed_features, total_feature_time)

        # 3. Build and log the report
        self._report_formatter.log_report(
            batch_size, total_time, total_own_waiting, total_own_processing,
            total_feature_time, processed_features, wait_candidates, optimize_candidates
        )
