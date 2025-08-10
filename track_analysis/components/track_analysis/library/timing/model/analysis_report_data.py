import dataclasses
from typing import List, Optional

from track_analysis.components.track_analysis.library.timing.model.parallelism_stats import ParallelismStats
from track_analysis.components.track_analysis.library.timing.model.processed_feature import ProcessedFeature
from track_analysis.components.track_analysis.library.timing.model.suggestion_categories import SuggestionCategories


@dataclasses.dataclass(frozen=True)
class AnalysisReportData:
    """A single data container for the entire timing and performance analysis report."""
    batch_size: int
    total_sequential_time: float
    total_own_waiting: float
    total_own_processing: float
    total_feature_time: float
    processed_features: List[ProcessedFeature]
    suggestions: SuggestionCategories
    parallelism_stats: Optional[ParallelismStats] = None
