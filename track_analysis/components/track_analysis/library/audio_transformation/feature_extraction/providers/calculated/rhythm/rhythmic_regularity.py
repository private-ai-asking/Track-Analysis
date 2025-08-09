from pathlib import Path
from typing import List, Dict, Any

import librosa
import numpy as np

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.shared.caching.hdf5_memory import TimedCacheResult
# Assuming your existing imports and class definitions are available
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.timed_cache(identifier_arg="file_path", ignore=["onset_envelope"])
def _compute_rhythmic_regularity(
        *,
        file_path: Path,
        onset_envelope: np.ndarray,
        unique_string: str,
        algorithm_version: str = "v2"
) -> TimedCacheResult[float]:
    """
    Cached calculation for rhythmic regularity using autocorrelation.
    Includes normalization to ensure the result is scaled correctly.
    """
    if onset_envelope.size < 2:
        return 0.0 # type: ignore

    # Calculate the autocorrelation of the onset envelope
    autocorr = librosa.autocorrelate(onset_envelope)

    energy = autocorr[0]
    if energy < 1e-9:
        return 0.0 # type: ignore

    normalized_autocorr = autocorr / energy

    # Exclude the first peak (at lag 0, which is now 1.0)
    # and find the maximum of the rest of the autocorrelation function.
    if len(normalized_autocorr) > 1:
        regularity = np.max(normalized_autocorr[1:])
    else:
        regularity = 0.0

    return float(regularity) # type: ignore


class RhythmicRegularityProvider(AudioDataFeatureProvider):
    """
    Provides a measure of rhythmic regularity based on the autocorrelation
    of the percussive onset envelope.
    High values indicate a steady, predictable rhythmic pattern.
    Low values indicate an irregular or non-existent rhythm.
    """
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.AUDIO_PATH,
            AudioDataFeature.PERCUSSIVE_ONSET_ENVELOPE,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.RHYTHMIC_REGULARITY

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            file_path = data[AudioDataFeature.AUDIO_PATH]
            percussive_onset_env = data[AudioDataFeature.PERCUSSIVE_ONSET_ENVELOPE]

        regularity = _compute_rhythmic_regularity(
            file_path=file_path,
            onset_envelope=percussive_onset_env,
            unique_string="percussive_rhythmic_regularity"
        )
        self._add_timed_cache_times(regularity)

        return {AudioDataFeature.RHYTHMIC_REGULARITY: regularity.value}
