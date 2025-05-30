import numpy as np


class SequenceOptimizer:
    def __init__(self, transition_penalty: float):
        """
        penalty for switching states; staying has zero cost
        """
        self._penalty = transition_penalty

    def solve(self, scores: np.ndarray) -> np.ndarray:
        n, m = scores.shape
        dp = np.full((n, m), -np.inf)
        backptr = np.zeros((n, m), dtype=int)
        dp[0, :] = scores[0, :]
        for i in range(1, n):
            for j in range(m):
                stay = dp[i - 1, j]
                switch = np.max(dp[i - 1, :] - self._penalty)
                best_prev = j if stay >= switch else int(np.argmax(dp[i - 1, :] - self._penalty))
                dp[i, j] = scores[i, j] + max(stay, switch)  # type: ignore
                backptr[i, j] = best_prev
        # backtrace
        path = np.zeros(n, dtype=int)
        path[-1] = int(np.argmax(dp[-1, :]))
        for i in range(n - 2, -1, -1):
            path[i] = backptr[i + 1, path[i + 1]]
        return path
