from typing import NamedTuple, List

from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler


class EnergyModel(NamedTuple):
    scaler: RobustScaler
    pca: PCA
    spline: CubicSpline
    feature_names: List[str]
    spline_y_points: List[float]
    data_hash: str
    features_shape: List[int]
