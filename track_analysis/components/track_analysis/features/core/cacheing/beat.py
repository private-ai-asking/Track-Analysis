from pathlib import Path
import numpy as np
import librosa.beat

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.core.cacheing.harmonic import HarmonicExtractor
from track_analysis.components.track_analysis.features.core.cacheing.shared import MEMORY


@MEMORY.cache(ignore=["audio"])
def _compute_beat_track(
        *,
        file_path: Path,
        start_sample: int,
        end_sample: int,
        sample_rate: int,
        audio: np.ndarray = None
) -> tuple[float, np.ndarray]:
    """
    Cached beat tracking:
    - Cache key: (file_path, start_sample, end_sample, sample_rate)
    - `audio` is ignored for the cache key but used if provided.
    """
    # Slice or memmap the audio segment
    if audio is None:
        length = end_sample - start_sample
        audio = np.memmap(
            str(file_path), dtype="float32", mode="r",
            offset=start_sample * 4,
            shape=(length,)
        )
    else:
        audio = audio[start_sample:end_sample]

    # Run beat tracking
    tempo, frames = librosa.beat.beat_track(
        y=audio,
        sr=sample_rate,
        units='frames',
        trim=False
    )
    return float(tempo), np.array(frames)


class BeatDetector:
    """
    Detects beats and estimates tempo from audio, with optimized disk cache.
    """
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._hp_extractor = HarmonicExtractor(logger)

    def detect(
            self,
            *,
            audio_path: Path,
            audio: np.ndarray,
            sample_rate: int
    ) -> tuple[float, np.ndarray, np.ndarray]:
        # ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # extract percussive component, cached via HarmonicExtractor
        percussive = self._hp_extractor.percussive_only(
            file_path=audio_path,
            audio=audio,
            start_sample=0,
            end_sample=audio.shape[0]
        )

        # cached beat track on file+range
        tempo, frames = _compute_beat_track(
            file_path=audio_path,
            start_sample=0,
            end_sample=percussive.shape[0],
            sample_rate=sample_rate,
            audio=percussive
        )

        # convert frame indices to times
        times = librosa.frames_to_time(frames, sr=sample_rate)

        self._logger.info(
            f"Estimated tempo: {tempo:.2f} BPM", separator=self._separator
        )
        return tempo, frames, times
