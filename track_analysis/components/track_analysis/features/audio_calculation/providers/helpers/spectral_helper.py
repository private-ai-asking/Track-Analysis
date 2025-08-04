from functools import lru_cache

import numpy as np
from numba import njit, prange


# noinspection t
@njit(parallel=True)
def spectral_centroid_and_flux(
        S: np.ndarray,
        freqs: np.ndarray,
        cent_out: np.ndarray,
        flux_out: np.ndarray
) -> None:
    """
    Compute spectral centroid and flux in-place, in parallel using Numba.
    """
    n_bins, n_frames = S.shape
    for j in prange(n_frames):
        # Spectral Centroid
        num = 0.0
        den = 1e-8
        for i in range(n_bins):
            v = S[i, j]
            num += freqs[i] * v
            den += v
        cent_out[j] = num / den

        # Spectral Flux
        if j > 0:
            acc = 0.0
            for i in range(n_bins):
                d = S[i, j] - S[i, j - 1]
                if d > 0:
                    acc += d * d
            flux_out[j - 1] = acc ** 0.5

@lru_cache(maxsize=None)
def spectral_get_freqs(sr: int, n_bins: int) -> np.ndarray:
    """
    Calculate and cache the frequency values for each bin of an FFT.
    """
    return np.linspace(0, sr / 2, n_bins)
