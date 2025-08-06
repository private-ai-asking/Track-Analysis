from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
from joblib import Memory
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.utils.convert_to_db import convert_to_db_func
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.feature.visualization.renderers.base_renderer import \
    Renderer


def _compute_fft(audio: np.ndarray, fft: int, hop_length: int) -> np.ndarray:
    D = librosa.stft(audio, n_fft=fft, hop_length=hop_length)
    S = np.abs(D)
    return S

def _compute_time_and_frequencies(data: np.ndarray, hop_length: int, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    n_freq, n_frame = data.shape
    times = np.arange(n_frame) * hop_length / sample_rate
    freqs = np.linspace(0, sample_rate / 2, n_freq)
    return times, freqs

class SpectrogramRenderer(Renderer):
    def __init__(
            self,
            cache_dir: Path,
            sample_rate: int,
            hop_length: int,
            to_db: bool = True,
            ref: float = 1.0,
            cmap: str = "magma",
            n_fft: Optional[int] = None,
    ):
        self._compute_db = convert_to_db_func
        self._compute_fft = Memory(cache_dir, verbose=0).cache(_compute_fft)
        self._compute_times_and_frequencies = Memory(cache_dir, verbose=0).cache(_compute_time_and_frequencies)

        self._sr      = sample_rate
        self._hop     = hop_length
        self._to_db   = to_db
        self._ref     = ref
        self._cmap    = cmap
        self._n_fft   = n_fft or 2048

    # noinspection PyTupleAssignmentBalance
    def render(self, ax: plt.Axes, data: np.ndarray) -> Optional[plt.cm.ScalarMappable]:
        # --- 1) Turn 1D audio into a magnitude spectrogram if needed ---
        if data.ndim == 1:
            S = self._compute_fft(data, self._n_fft, self._hop)
        elif data.ndim == 2:
            S = data
        else:
            raise ValueError(f"Expected 1D audio or 2D spectrogram, got shape {data.shape}")

        # --- 2) Optional dB conversion ---
        if self._to_db:
            S = self._compute_db(S, self._ref)

        # --- 3) Compute time & freq axes ---
        times, freqs = self._compute_times_and_frequencies(S, self._hop, self._sr)
        ax.set_yscale("log")
        ax.set_ylim(freqs[1], freqs[-1])
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{int(x)}"))

        # --- 4) Draw and return the mesh for the colorbar ---
        mesh = ax.pcolormesh(times, freqs, S, shading="auto", cmap=self._cmap)

        return mesh
