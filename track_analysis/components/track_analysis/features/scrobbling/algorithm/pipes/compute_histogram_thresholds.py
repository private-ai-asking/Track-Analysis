from typing import Tuple, Optional

from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.scrobbling.algorithm.algorithm_context import AlgorithmContext
import numpy as np


class ComputeHistogramThresholds(IPipe):
    """
    Compute two dynamic thresholds (reject/accept) by performing multi-level Otsu thresholding
    on 1D confidence scores via histogram maximization of between-class variance.
    """

    def __init__(
            self,
            logger: HoornLogger,
            n_classes: int = 3,
            nbins: int = 128,
            random_state: int = 0,
    ):
        self._logger = logger
        self._separator = "CacheBuilder.ComputeHistogramThresholds"
        self._n_classes = n_classes
        self._nbins = nbins
        self._random_state = random_state
        self._logger.trace(
            f"Initialized ComputeHistogramThresholds with {n_classes} classes and {nbins} bins.",
            separator=self._separator,
        )

    def flow(self, ctx: AlgorithmContext) -> AlgorithmContext:
        confidences = self._gather_confidences(ctx)
        if confidences.size < self._n_classes:
            self._logger.error(
                "Insufficient data for histogram thresholding.",
                separator=self._separator,
            )
            return ctx

        thr_reject, thr_accept = self._compute_histogram_thresholds(confidences.ravel())
        if thr_reject is None or thr_accept is None:
            self._logger.error(
                "Failed to compute histogram-based thresholds.",
                separator=self._separator,
            )
            return ctx

        ctx.dynamic_reject_threshold = thr_reject
        ctx.dynamic_accept_threshold = thr_accept
        self._logger.info(
            f"Derived dynamic thresholds: reject={thr_reject:.3f}, accept={thr_accept:.3f}",
            separator=self._separator,
        )
        return ctx

    def _gather_confidences(self, ctx: AlgorithmContext) -> np.ndarray:
        parts = []
        df = getattr(ctx, 'scrobble_data_frame', None)
        if df is not None and not df.empty and '__confidence' in df.columns:
            parts.append(df['__confidence'].values)
        for attr in ('auto_accepted_scrobbles', 'auto_rejected_scrobbles', 'confused_scrobbles'):
            df = getattr(ctx, attr, None)
            if df is not None and not df.empty and '__confidence' in df.columns:
                parts.append(df['__confidence'].values)
        if not parts:
            return np.empty((0,))
        all_conf = np.concatenate(parts)
        # subsample up to 10k points for speed
        N = all_conf.shape[0]
        max_s = 10_000
        if N > max_s:
            rng = np.random.default_rng(self._random_state)
            idx = rng.choice(N, size=max_s, replace=False)
            all_conf = all_conf[idx]
        return all_conf

    def _compute_histogram_thresholds(
            self,
            data: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Brute-force multi-Otsu on 1D data via histogram.
        Returns two thresholds splitting data into three classes that maximize
        between-class variance. Complexity: O(nbins^2).
        """
        if data.size == 0:
            return None, None

        counts, bin_edges = np.histogram(data, bins=self._nbins)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        total = counts.sum()
        global_mean = (counts * centers).sum() / total

        best_score = -np.inf
        best_t1 = best_t2 = None

        # precompute cumulative sums for fast range sums
        c_counts = np.cumsum(counts)
        c_weighted = np.cumsum(counts * centers)

        for i in range(1, self._nbins - 1):
            w0 = c_counts[i - 1]
            mu0 = c_weighted[i - 1] / w0 if w0 > 0 else 0
            for j in range(i + 1, self._nbins):
                w1 = c_counts[j - 1] - c_counts[i - 1]
                mu1 = ((c_weighted[j - 1] - c_weighted[i - 1]) / w1) if w1 > 0 else 0
                w2 = total - c_counts[j - 1]
                mu2 = ((c_weighted[-1] - c_weighted[j - 1]) / w2) if w2 > 0 else 0

                score = (
                        w0 * (mu0 - global_mean) ** 2
                        + w1 * (mu1 - global_mean) ** 2
                        + w2 * (mu2 - global_mean) ** 2
                )
                if score > best_score:
                    best_score = score
                    best_t1 = centers[i]
                    best_t2 = centers[j]

        return best_t1, best_t2
