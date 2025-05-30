from abc import ABC, abstractmethod

import numpy as np


class FeatureTransformer(ABC):
    @abstractmethod
    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw per-segment data into a feature vector."""
        pass
