from dataclasses import dataclass
from typing import List

from track_analysis.components.track_analysis.features.data_generation.model.header import Header


@dataclass(frozen=True)
class EnergyModelConfig:
    name: str
    feature_columns: List[Header]
