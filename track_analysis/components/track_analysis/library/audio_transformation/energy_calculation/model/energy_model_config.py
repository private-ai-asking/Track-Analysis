from dataclasses import dataclass
from typing import List

from track_analysis.components.track_analysis.features.data_generation.model.header import Header


@dataclass(frozen=True)
class EnergyModelConfig:
    name: str
    feature_columns: List[Header | str]
    use_mfcc: bool = True

    def get_feature_names(self) -> List[str]:
        feature_names: List[str] = []
        for column in self.feature_columns:
            if isinstance(column, Header):
                feature_names.append(column.value)
                continue
            elif isinstance(column, str):
                feature_names.append(column)
                continue
            else: raise ValueError("Unknown column type")

        return feature_names
