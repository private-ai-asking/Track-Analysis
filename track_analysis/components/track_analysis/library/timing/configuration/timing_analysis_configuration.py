import dataclasses
from typing import List


@dataclasses.dataclass(frozen=True)
class TimingAnalysisConfiguration:
    timing_data_name: str = "Run"
    """This is the name for the timing data."""

    timing_data_name_plural: str = "Runs"
    """This is the name for the timing data in plural."""

    minimum_impact_percentage_threshold: float = 5
    """A feature must have at least this percentage of impact on the total execution for it to be considered as a suggestion for improvements."""

    minimum_total_time_spent_threshold: float = 0.1
    """A feature must have spent at least this amount of time (in seconds) executing for it to be considered as a suggestion for improvements."""

    minimum_spent_ratio: float = 0.6
    """A feature must have spent this amount of time relative to the total in X state (processing or waiting) for it to be considered as a suggestion for improvements."""

    variance_threshold: float = 0.5
    """A feature's standard deviation must be this fraction of its average time to be flagged."""

    caching_time_threshold: float = 0.5
    """A feature must run longer than this (in seconds) to be considered for caching suggestions."""

    ignore_suggestions_for: List[str] = dataclasses.field(default_factory=list)
    """A list of feature names to exclude from all suggestions."""
