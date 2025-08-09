import dataclasses


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
