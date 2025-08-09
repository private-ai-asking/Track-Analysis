from typing import NamedTuple, List

from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

class TrainingShape(NamedTuple):
    training_samples: int
    number_of_features: int

    def to_dict(self) -> dict:
        return {
            "training_samples": self.training_samples,
            "number_of_features": self.number_of_features,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingShape":
        return cls(
            training_samples=d["training_samples"],
            number_of_features=d["number_of_features"],
        )


class EnergyModel(NamedTuple):
    scaler: RobustScaler
    pca: PCA
    spline: CubicSpline
    feature_names: List[str]
    spline_y_points: List[float]
    data_hash: str
    features_shape: TrainingShape
    number_of_pca_components: int
    cumulative_variance: float
