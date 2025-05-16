from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.scrobbling.algorithm.algorithm_context import AlgorithmContext


class ComputeGMMThresholds(IPipe):
    """
    Fit a 3-component GMM to confidence scores and derive two cut-points:
    1. reject_threshold between rejects and uncertain
    2. accept_threshold between uncertain and accepts
    """

    def __init__(
            self,
            logger: HoornLogger,
            n_components: int = 3,
            random_state: int = 0
    ):
        self._logger = logger
        self._separator = "CacheBuilder.ComputeGMMThresholds"
        self._n_components = n_components
        self._random_state = random_state
        self._logger.trace(
            "Initialized ComputeGMMThresholds.",
            separator=self._separator
        )

    def flow(self, ctx: AlgorithmContext) -> AlgorithmContext:
        confidences = self._gather_confidences(ctx)
        if confidences.size < self._n_components:
            self._logger.error(
                "Insufficient data for GMM fitting.",
                separator=self._separator
            )
            return ctx

        gmm = self._fit_gmm(confidences)
        means, covariances = self._sort_components(gmm)
        thr_reject, thr_accept = self._compute_thresholds(means, covariances)

        if thr_reject is None or thr_accept is None:
            self._logger.error(
                "Failed to compute dynamic thresholds.",
                separator=self._separator
            )
            return ctx

        ctx.dynamic_reject_threshold = thr_reject
        ctx.dynamic_accept_threshold = thr_accept
        self._logger.info(
            f"Derived dynamic thresholds: reject={thr_reject:.3f}, accept={thr_accept:.3f}",
            separator=self._separator
        )
        return ctx

    def _gather_confidences(self, ctx: AlgorithmContext) -> np.ndarray:
        """
        Reconstruct a 1D numpy array of all __confidence values
        from context, whether in scrobble_data_frame or split pipelines.
        Subsamples up to 10k points for faster GMM fitting.
        """
        parts = []
        # Collect from main scrobble dataframe
        df = getattr(ctx, 'scrobble_data_frame', None)
        if df is not None and not df.empty and '__confidence' in df.columns:
            parts.append(df['__confidence'].values)

        # Collect from buckets
        for attr in ('auto_accepted_scrobbles', 'auto_rejected_scrobbles', 'confused_scrobbles'):
            df = getattr(ctx, attr, None)
            if df is not None and not df.empty and '__confidence' in df.columns:
                parts.append(df['__confidence'].values)

        if not parts:
            return np.empty((0, 1), dtype=float)

        all_conf = np.concatenate(parts)
        # Sub-sample if too large
        N = all_conf.shape[0]
        max_samples = 10_000
        if N > max_samples:
            rng = np.random.default_rng(self._random_state)
            idx = rng.choice(N, size=max_samples, replace=False)
            sampled = all_conf[idx]
        else:
            sampled = all_conf

        return sampled.reshape(-1, 1)

    def _fit_gmm(self, data: np.ndarray) -> GaussianMixture:
        """Instantiate and fit a lightweight GMM model to the data."""
        model = GaussianMixture(
            n_components=self._n_components,
            covariance_type='diag',  # cheaper for 1D
            max_iter=20,            # fewer EM iterations
            tol=1e-3,               # early stopping threshold
            n_init=1,               # single initialization
            random_state=self._random_state,
        )
        model.fit(data)
        return model

    def _sort_components(
            self, gmm: GaussianMixture
    ) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        """Return sorted means and covariances by ascending mean values."""
        means = gmm.means_.flatten()
        covs = gmm.covariances_.flatten()
        order = np.argsort(means)
        return means[order], covs[order]

    def _compute_intersection(
            self,
            mean1: float,
            var1: float,
            mean2: float,
            var2: float
    ) -> Optional[float]:
        # unchanged
        a = 1/(2 * var1) - 1/(2 * var2)
        b = mean2/var2 - mean1/var1
        c = (mean1**2)/(2 * var1) - (mean2**2)/(2 * var2) - \
            np.log(np.sqrt(var2 / var1))
        disc = b**2 - 4 * a * c
        if disc < 0:
            return None
        roots = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
        return next(
            (r for r in roots if min(mean1, mean2) < r < max(mean1, mean2)),
            None
        )

    def _compute_thresholds(
            self,
            means: np.ndarray[float],
            covariances: np.ndarray[float]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute two thresholds: between components 0-1 and 1-2.
        """
        thr_reject = self._compute_intersection(
            means[0], covariances[0], means[1], covariances[1]
        )
        thr_accept = self._compute_intersection(
            means[1], covariances[1], means[2], covariances[2]
        )
        return thr_reject, thr_accept
