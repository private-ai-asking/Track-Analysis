import dataclasses


@dataclasses.dataclass(frozen=True)
class FeatureTimingData:
    associated_feature: str
    """The associated feature's name."""

    time_spent_waiting: float
    """The time the feature spent waiting."""

    time_spent_processing: float
    """The time the feature spent processing."""
