from pathlib import Path
from typing import Dict, Any, List, Tuple

import librosa
import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.separation.calculator.harmonic import \
    HarmonicExtractor
from track_analysis.components.track_analysis.shared.caching.hdf5_memory import TimedCacheResult
from track_analysis.components.track_analysis.shared_objects import MEMORY
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider

@MEMORY.timed_cache(identifier_arg="file_path", ignore=["audio", "onset_envelope", "tempo"])
def _compute_beat_track(
        *,
        file_path: Path,
        start_sample: int,
        end_sample: int,
        sample_rate: int,
        audio: np.ndarray = None,
        onset_envelope: np.ndarray = None,
        hop_length: int = 512,
        tempo: float = None,
) -> TimedCacheResult[tuple[float, np.ndarray]]:
    """
    Cached beat tracking:
    - Cache key: (file_path, start_sample, end_sample, sample_rate)
    """
    audio = audio[start_sample:end_sample]

    # Run beat tracking
    tempo, frames = librosa.beat.beat_track(
        y=audio,
        sr=sample_rate,
        units='frames',
        trim=False,
        hop_length=hop_length,
        onset_envelope=onset_envelope,
        bpm=tempo,
    )
    return float(tempo), np.array(frames) # type: ignore


# TODO - Consolidate into the tempo provider once key extraction has been integrated as feature provider.
class BeatDetector:
    """
    Detects beats and estimates tempo from audio, with optimized disk cache.
    """
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._hp_extractor = HarmonicExtractor(logger)

    def get_tempo(
            self,
            *,
            audio_path: Path,
            audio: np.ndarray,
            sample_rate: int,
            onset_envelope: np.ndarray | None,
            hop_length: int = 512,
    ) -> TimedCacheResult[float]:
        # NOTE TO SELF: I actually must process a different percussive component here
        # because the HPS provider depends on the computed tempo to separate.
        percussive = self._pre_process(audio, audio_path)

        results = _compute_beat_track(
            file_path=audio_path,
            start_sample=0,
            end_sample=len(percussive.value),
            sample_rate=sample_rate,
            onset_envelope=onset_envelope,
            hop_length=hop_length,
            audio=audio,
        )

        tempo, _ = results.value

        self._logger.info(
            f"Estimated tempo: {tempo:.2f} BPM", separator=self._separator
        )

        return TimedCacheResult(
            value=tempo,
            time_processing=percussive.time_processing+results.time_processing,
            time_waiting=results.time_waiting+results.time_waiting,
            retrieved_from_cache=results.retrieved_from_cache,
        )

    def get_beat_frames_and_times(
            self,
            *,
            audio_path: Path,
            audio: np.ndarray,
            sample_rate: int,
            onset_envelope: np.ndarray | None,
            hop_length: int = 512,
            tempo: float = None,
    ) -> TimedCacheResult[Tuple[np.ndarray, np.ndarray]]:
        """
        Set tempo if you pre-computed through get_tempo. This might speed up the process (untested)
        """

        # NOTE TO SELF: I actually must process a different percussive component here
        # because the HPS provider depends on the computed tempo to separate.
        percussive = self._pre_process(audio, audio_path)

        results = _compute_beat_track(
            file_path=audio_path,
            start_sample=0,
            end_sample=len(percussive.value),
            sample_rate=sample_rate,
            onset_envelope=onset_envelope,
            hop_length=hop_length,
            audio=audio,
            tempo=tempo,
        )

        _, frames = results.value

        times = librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length)

        return TimedCacheResult(
            value=(frames, times),
            time_processing=percussive.time_processing+results.time_processing,
            time_waiting=results.time_waiting+results.time_waiting,
            retrieved_from_cache=results.retrieved_from_cache,
        )

    def _pre_process(self, audio: np.ndarray, audio_path: Path) -> TimedCacheResult[np.ndarray]:
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        percussive = self._hp_extractor.percussive_only(
            file_path=audio_path,
            audio=audio,
            start_sample=0,
            end_sample=audio.shape[0]
        )

        return percussive


class TempoProvider(AudioDataFeatureProvider):
    def __init__(self, beat_detector: BeatDetector, hop_length: int):
        super().__init__()
        self._beat_detector = beat_detector
        self._hop_length = hop_length

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.AUDIO_PATH, AudioDataFeature.AUDIO_SAMPLES, AudioDataFeature.SAMPLE_RATE_HZ, AudioDataFeature.ONSET_ENVELOPE]

    @property
    def output_features(self) -> AudioDataFeature | List[AudioDataFeature]:
        return AudioDataFeature.BPM

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            audio_path = data[AudioDataFeature.AUDIO_PATH]
            samples = data[AudioDataFeature.AUDIO_SAMPLES]
            sample_rate = data[AudioDataFeature.SAMPLE_RATE_HZ]
            onset_envelope = data[AudioDataFeature.ONSET_ENVELOPE]

        tempo = self._beat_detector.get_tempo(audio_path=audio_path, audio=samples, sample_rate=sample_rate, onset_envelope=onset_envelope, hop_length=self._hop_length)
        self._add_timed_cache_times(tempo)

        with self._measure_processing():
            return {
                AudioDataFeature.BPM: tempo.value,
            }
