from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.shared.caching.hdf5_memory import TimedCacheResult
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.timed_cache(identifier_arg="file_path", ignore=["beat_times", "beat_frames"])
def generate_subbeat_events(
        file_path: Path,
        subdivisions: int,
        beat_times: np.ndarray,
        beat_frames: np.ndarray) -> TimedCacheResult[List[Tuple[float, int]]]:
    events: List[Tuple[float, int]] = []

    for i in range(beat_times.size - 1):
        t0, t1 = beat_times[i], beat_times[i + 1]
        f0, f1 = beat_frames[i], beat_frames[i + 1]
        interval = t1 - t0
        frame_span = f1 - f0

        for sub in range(subdivisions):
            frac = sub / subdivisions
            time_point = t0 + frac * interval
            frame_point = int(f0 + frac * frame_span)
            events.append((time_point, frame_point))  # type: ignore

    events.append((float(beat_times[-1]), int(beat_frames[-1])))
    return sorted(events, key=lambda evt: evt[0]) # type: ignore

class SubBeatEventsProvider(AudioDataFeatureProvider):
    def __init__(self, subdivisions_per_beat: int):
        super().__init__()
        self._subdivisions_per_beat = subdivisions_per_beat

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.BEAT_TIMES, AudioDataFeature.BEAT_FRAMES]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return [AudioDataFeature.SUB_BEAT_EVENTS]

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            audio_path = data[AudioDataFeature.AUDIO_PATH]
            beat_times = data[AudioDataFeature.BEAT_TIMES]
            beat_frames = data[AudioDataFeature.BEAT_FRAMES]

        sub_beat_events_result = generate_subbeat_events(
            file_path=audio_path,
            subdivisions=self._subdivisions_per_beat,
            beat_times=beat_times,
            beat_frames=beat_frames
        )
        self._add_timed_cache_times(sub_beat_events_result)

        with self._measure_processing():
            return {
                AudioDataFeature.SUB_BEAT_EVENTS: sub_beat_events_result.value
            }
