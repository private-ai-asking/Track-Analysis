import os
from pathlib import Path
from typing import Optional, Literal, Tuple

import numpy as np
from joblib import Memory
from matplotlib import pyplot as plt

from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.feature.visualization.renderers.base_renderer import \
    Renderer

def _arrange_times(num_frames: int, sample_rate: int):
    return np.arange(num_frames) / sample_rate

def _downsample(times: np.ndarray, data: np.ndarray, num_frames: int, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.linspace(0, num_frames - 1, max_points, dtype=int)
    times = times[indices]
    data = data[indices]

    return times, data

def _compute_rms_envelope(
        data: np.ndarray,
        sample_rate: int,
        frame_length_s: float = 0.1,
        hop_factor: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (times, rms) for a short‐time RMS envelope
    of `data`, computed with a window of `frame_length_s` seconds
    and a hop of `hop_factor * frame_length_s`.
    """
    frame_len = int(frame_length_s * sample_rate)
    hop      = int(frame_len * hop_factor)

    padded = np.pad(data**2, (frame_len//2, frame_len//2), mode="reflect")
    window = np.ones(frame_len) / frame_len
    rms    = np.sqrt(np.convolve(padded, window, mode="valid")[::hop])

    times  = (np.arange(rms.size) * hop + frame_len/2) / sample_rate

    return times, rms


class WaveformRenderer(Renderer):
    """Render a 1D audio waveform in multiple modes using Matplotlib."""
    def __init__(
            self,
            cache_dir: Path,
            sample_rate: int,
            mode: Literal["Full", "Envelope", "Min/Max"] = "Envelope",
            max_points: Optional[int] = None
    ):
        os.makedirs(cache_dir, exist_ok=True)
        self._downsample = Memory(cache_dir, verbose=0).cache(_downsample)
        self._arrange_times = Memory(cache_dir, verbose=0).cache(_arrange_times)
        self._compute_rms_envelope = Memory(cache_dir, verbose=0).cache(_compute_rms_envelope)

        self._sample_rate = sample_rate
        self._mode = mode
        self._max_points = max_points or self._sample_rate

    def render(self, ax: plt.Axes, data: np.ndarray) -> None:
        """Dispatch to the selected rendering mode."""
        if self._mode == "Full":
            self._render_full(ax, data)
        elif self._mode == "Envelope":
            self._render_envelope(ax, data)
        elif self._mode == "Min/Max":
            self._render_min_max(ax, data)
        else:
            raise ValueError(f"Unknown mode: {self._mode}")
        return None

    def _render_full(self, ax: plt.Axes, data: np.ndarray) -> None:
        """Plot every sample as a continuous line (downsampled if needed)."""
        num_frames = data.shape[0]
        times = self._arrange_times(num_frames, self._sample_rate)
        if num_frames > self._max_points:
            times, data = self._downsample(times, data, num_frames, self._max_points)

        ax.plot(times, data, linewidth=0.8)

    def _render_envelope(self, ax: plt.Axes, data: np.ndarray) -> None:
        """Compute and plot a short-time RMS amplitude envelope."""
        times, rms = self._compute_rms_envelope(data, self._sample_rate, frame_length_s=0.1, hop_factor=0.5)
        ax.plot(times, rms, linewidth=1.0)

    def _render_min_max(self, ax: plt.Axes, data: np.ndarray) -> None:
        """Aggregate min/max per block and render a filled region."""
        num_frames = data.shape[0]
        # choose number of blocks = max_points
        blocks = min(self._max_points, num_frames)
        block_size = int(np.ceil(num_frames / blocks))
        mins = []
        maxs = []
        times = []
        for i in range(0, num_frames, block_size):
            block = data[i:i + block_size]
            mins.append(block.min())
            maxs.append(block.max())
            times.append((i + block_size/2) / self._sample_rate)
        times = np.array(times)
        ax.fill_between(times, mins, maxs, linewidth=0, alpha=0.6)
