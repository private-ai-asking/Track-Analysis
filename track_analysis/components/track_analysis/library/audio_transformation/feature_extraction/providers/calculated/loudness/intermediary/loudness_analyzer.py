import math
from pathlib import Path
from typing import List, Tuple, Any, Dict

import numpy as np
from pyebur128 import (
    R128State,
    MeasurementMode,
    get_loudness_shortterm,
    get_loudness_range,
    get_true_peak, get_loudness_global,
)

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.loudness.intermediary.loudness_analysis_result import \
    LoudnessAnalysisResult
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.cache(identifier_arg="file_path", ignore=["audio"])
def _compute_shortterm_lufs_array(
        *,
        file_path: Path,
        sample_rate: int,
        chunk_size: int,
        hop_size: int,
        audio: np.ndarray,
) -> np.ndarray:
    """
    Returns a NumPy array of short-term LUFS values (one per hop).
    Cached on (file_path, sample_rate, chunk_size, hop_size) only.
    """
    # ensure 2D: (frames, channels)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    frames, channels = audio.shape

    st_s = R128State(channels, sample_rate, MeasurementMode.MODE_S)
    shortterm_values: List[float] = []

    for off in range(0, frames, hop_size):
        block = audio[off : off + chunk_size]
        flat  = block.flatten()
        n     = block.shape[0]

        st_s.add_frames(flat, n)
        try:
            shortterm_values.append(get_loudness_shortterm(st_s))
        except ValueError:
            pass

    return np.array(shortterm_values, dtype=float)

@MEMORY.cache(identifier_arg="file_path", ignore=["audio"])
def _compute_global_loudness(
        *,
        file_path:   Path,
        sample_rate: int,
        chunk_size:  int,
        audio:       np.ndarray,
) -> Tuple[float, float, List[float], float, float, int]:
    """
    Returns (LUFS_I, LRA, TruePeak, peak, rms_all, channels)
    for the given audio. Cached on (file_path, sample_rate, chunk_size).
    """
    if audio.ndim == 1:
        audio = audio[:, None]
    frames, channels = audio.shape

    st_i   = R128State(channels, sample_rate, MeasurementMode.MODE_I)
    st_lra = R128State(channels, sample_rate, MeasurementMode.MODE_LRA)
    st_tp  = R128State(channels, sample_rate, MeasurementMode.MODE_TRUE_PEAK)

    sum_sq = 0.0
    max_sq = 0.0

    for off in range(0, frames, chunk_size):
        block = audio[off : off + chunk_size]
        flat  = block.flatten()
        n     = block.shape[0]

        for st in (st_i, st_lra, st_tp):
            st.add_frames(flat, n)

        sq     = block**2
        sum_sq += float(sq.sum())
        max_sq  = max(max_sq, float(sq.max()))

    peak    = math.sqrt(max_sq)
    rms_all = math.sqrt(sum_sq / (frames * channels)) if frames > 0 else 0.0

    lufs_i = get_loudness_global(st_i)
    lra = get_loudness_range(st_lra)
    true_peak = [
        get_true_peak(st_tp, ch)
        for ch in range(0, channels)
    ]


    return lufs_i, lra, true_peak, peak, rms_all, channels


class LoudnessAnalyzer(AudioDataFeatureProvider):
    def __init__(self, chunk_size: int = 4096, hop_size: int = 512):
        self._chunk_size = chunk_size
        self._hop_size = hop_size

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.AUDIO_PATH,
            AudioDataFeature.AUDIO_SAMPLES,
            AudioDataFeature.SAMPLE_RATE_HZ,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.LOUDNESS_ANALYSIS_RESULT

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        fp      = data[AudioDataFeature.AUDIO_PATH]
        audio   = data[AudioDataFeature.AUDIO_SAMPLES]
        sr      = data[AudioDataFeature.SAMPLE_RATE_HZ]

        # 1) Global stats, cached
        lufs_i, lra, true_peak, peak, rms_all, ch = _compute_global_loudness(
            file_path=fp,
            sample_rate=sr,
            chunk_size=self._chunk_size,
            audio=audio,
        )

        # 2) Short-term array, cached
        shortterm = _compute_shortterm_lufs_array(
            file_path=fp,
            sample_rate=sr,
            chunk_size=self._chunk_size,
            hop_size=self._hop_size,
            audio=audio,
        )

        # CORRECTED: The LoudnessAnalysisResult class should be initialized with the final values
        return {
            AudioDataFeature.LOUDNESS_ANALYSIS_RESULT: LoudnessAnalysisResult(
                lufs_i=lufs_i,
                lra=lra,
                true_peak=true_peak,
                peak=peak,
                rms_all=rms_all,
                channels=ch,
                shortterm_lufs=shortterm,
            )
        }
