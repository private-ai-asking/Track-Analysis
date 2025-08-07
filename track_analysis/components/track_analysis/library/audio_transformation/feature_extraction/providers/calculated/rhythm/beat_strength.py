from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.cache(identifier_arg="file_path", ignore=["onset_envelope", "beat_frames"])
def _compute_beat_strength(
        *,
        file_path: Path,
        onset_envelope: np.ndarray,
        beat_frames: np.ndarray,
        unique_string: str,
) -> float:
    """
    Cached calculation for beat strength.
    """
    if beat_frames.size == 0:
        return 0.0

    # Ensure beat frames are within the bounds of the onset envelope
    valid_beat_frames = beat_frames[beat_frames < len(onset_envelope)]

    if valid_beat_frames.size == 0:
        return 0.0

    # The beat strength is the mean of the onset envelope's energy at each beat
    beat_strength = np.mean(onset_envelope[valid_beat_frames])
    return float(beat_strength)


class BeatStrengthProvider(AudioDataFeatureProvider):
    """
    Provides the beat strength (salience), which measures how prominent the main
    beat is by checking the onset energy at detected beat times.
    """
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.AUDIO_PATH,
            AudioDataFeature.ONSET_ENVELOPE,
            AudioDataFeature.BEAT_FRAMES,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.BEAT_STRENGTH

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        file_path = data[AudioDataFeature.AUDIO_PATH]
        onset_envelope = data[AudioDataFeature.ONSET_ENVELOPE]
        beat_frames = data[AudioDataFeature.BEAT_FRAMES]

        strength = _compute_beat_strength(
            file_path=file_path,
            onset_envelope=onset_envelope,
            beat_frames=beat_frames,
            unique_string="full_audio_beat_strength"
        )

        return {AudioDataFeature.BEAT_STRENGTH: strength}
