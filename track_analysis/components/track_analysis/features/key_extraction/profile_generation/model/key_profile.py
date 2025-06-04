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

    def geometric_median(self, tol=1e-6, max_iter=500, eps=1e-8):
        """
        Compute the geometric median of a set of points using Weiszfeld's algorithm.

        Parameters
        ----------
        tol : float, optional (default=1e-6)
            Convergence tolerance: stop when the L2-norm between successive estimates is below tol.
        max_iter : int, optional (default=500)
            Maximum number of iterations.
        eps : float, optional (default=1e-8)
            Small constant added to distances to avoid division by zero if the estimate coincides
            with one of the input points.

        Returns
        -------
        m : np.ndarray, shape (12,)
            The estimated geometric median of the input vectors.
        """
        # Convert input to a (N, 12) array if it isn't already
        pts = np.asarray(self.vectors, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 12:
            raise ValueError("Input must be an array of shape (N, 12).")

        # 1) Initialize y^(0) as the arithmetic mean
        y = np.mean(pts, axis=0)

        for _ in range(max_iter):
            # 2) Compute distances from current estimate to each point: shape (N,)
            diff = pts - y          # shape (N, 12)
            dists = np.linalg.norm(diff, axis=1) + eps  # (N,)   :contentReference[oaicite:2]{index=2}

            # 3) Compute weights: w_i = 1 / dists[i]
            w = 1.0 / dists        # shape (N,)

            # 4) Weighted average to get the next estimate
            y_next = (w[:, None] * pts).sum(axis=0) / w.sum()

            # 5) Check convergence
            if np.linalg.norm(y_next - y) < tol:
                return y_next

            y = y_next

        # If we exit loop without convergence, return last estimate
        return y

    def get_label(self) -> str:
        return f"{self.tonic} {self.mode}"
