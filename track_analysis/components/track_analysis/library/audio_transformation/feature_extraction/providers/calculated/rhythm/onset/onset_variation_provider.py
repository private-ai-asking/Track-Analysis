import numpy as np
import librosa
from typing import List, Dict, Any
from pathlib import Path

from track_analysis.components.track_analysis.shared.caching.hdf5_memory import TimedCacheResult
from track_analysis.components.track_analysis.shared_objects import MEMORY
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider


@MEMORY.timed_cache(identifier_arg="file_path", ignore=["onset_times"])
def _compute_onset_rate_variation(
        *,
        file_path: Path,
        onset_times: np.ndarray,
        total_duration_sec: float,
        window_sec: float,
        hop_sec: float,
        unique_string: str,
) -> TimedCacheResult[float]:
    """
    Cached calculation for the standard deviation of onset rates over time.
    """
    if len(onset_times) < 2:
        return 0.0 # type: ignore

    window_rates = []
    start_time = 0.0
    while start_time + window_sec <= total_duration_sec:
        end_time = start_time + window_sec

        # Count how many onsets fall within the current window
        onsets_in_window = np.sum((onset_times >= start_time) & (onset_times < end_time))

        # Calculate rate for this window (onsets per second)
        rate = onsets_in_window / window_sec
        window_rates.append(rate)

        # Move the window forward
        start_time += hop_sec

    if not window_rates:
        return 0.0 # type: ignore

    return np.std(window_rates) # type: ignore


class OnsetVariationProvider(AudioDataFeatureProvider):
    """
    Calculates the standard deviation of the onset rate over time.

    This feature measures rhythmic consistency. A low value suggests a steady,
    consistent rhythm or a lack of rhythm (ambient), while a high value
    suggests a rhythm that changes significantly throughout the track.
    """
    def __init__(self, window_sec: float = 2.0, hop_sec: float = 1.0):
        """
        Initializes the provider.

        Args:
            window_sec: The length of the analysis window in seconds.
            hop_sec: The amount to slide the window forward in seconds.
        """
        super().__init__()
        self._window_sec = window_sec
        self._hop_sec = hop_sec

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.AUDIO_PATH,
            AudioDataFeature.PERCUSSIVE_ONSET_PEAKS,
            AudioDataFeature.SAMPLE_RATE_HZ,
            AudioDataFeature.AUDIO_SAMPLES,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_RATE_VARIATION

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            onset_frames = data[AudioDataFeature.PERCUSSIVE_ONSET_PEAKS]
            sample_rate = data[AudioDataFeature.SAMPLE_RATE_HZ]
            total_duration_sec = len(data[AudioDataFeature.AUDIO_SAMPLES]) / sample_rate
            file_path = data[AudioDataFeature.AUDIO_PATH]

            onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)

        onset_rate_variation = _compute_onset_rate_variation(
            file_path=file_path,
            onset_times=onset_times,
            total_duration_sec=total_duration_sec,
            window_sec=self._window_sec,
            hop_sec=self._hop_sec,
            unique_string="percussive_onset_variation"
        )
        self._add_timed_cache_times(onset_rate_variation)

        with self._measure_processing():
            return {AudioDataFeature.ONSET_RATE_VARIATION: onset_rate_variation.value}
