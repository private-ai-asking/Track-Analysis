from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class KeyProfile:
    tonic: str
    mode: str
    vectors: List[np.ndarray]

    def median_vector(self) -> np.ndarray:
        """
        Compute and return the component-wise median of all vectors in this profile.

        Assumes:
            - All arrays in `self.vectors` are 1-D and have the same length.
            - No NaNs or shape/dtype inconsistencies exist.

        Returns:
            A 1-D NumPy array of the same length as each element in `self.vectors`.
        """
        stacked: np.ndarray = np.stack(self.vectors)
        return np.median(stacked, axis=0)

    def get_label(self) -> str:
        return f"{self.tonic} {self.mode}"
