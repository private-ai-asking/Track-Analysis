import time
from pathlib import Path
import numpy as np
import librosa.effects

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.shared.caching.hdf5_memory import TimedCacheResult
from track_analysis.components.track_analysis.shared_objects import MEMORY


# TODO - Consolidate into HPSS feature provider once key extraction has been integrated into feature provider.

@MEMORY.timed_cache(identifier_arg="file_path", ignore=["audio"])
def _compute_hpss(
        *,
        file_path: Path,
        start_sample: int,
        end_sample: int,
        hop_length: int = 512,
        n_fft: int = 2048,
        kernel_size: int = None,
        margin: float = None,
        audio: np.ndarray = None,
) -> TimedCacheResult[tuple[np.ndarray, np.ndarray]]:
    """
    Cached HPSS computation:
    - Cache key: (file_path, start_sample, end_sample, hop_length, n_fft, kernel_size, margin)
    """
    audio = audio[start_sample:end_sample]

    # Perform harmonic-percussive source separation
    if margin is not None:
        harmonic, percussive = librosa.effects.hpss(
            y=audio,
            margin=margin
        )
    else:
        harmonic, percussive = librosa.effects.hpss(
            y=audio,
            kernel_size=kernel_size,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=None,
            power=2
        )

    return harmonic, percussive # type: ignore


class HarmonicExtractor:
    def __init__(
            self,
            logger: HoornLogger,
            hop_length: int = 512,
            n_fft: int = 2048
    ):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._hop_length = hop_length
        self._n_fft = n_fft
        self._logger.trace("HarmonicExtractor initialized.", separator=self._separator)

    def percussive_only(
            self,
            *,
            file_path: Path = None,
            audio: np.ndarray = None,
            start_sample: int = 0,
            end_sample: int = None,
            margin: float = 1.0
    ) -> TimedCacheResult[np.ndarray]:
        """
        Extracts only the percussive component, using cache when file_path is given.
        """
        if end_sample is None and audio is not None:
            end_sample = audio.shape[0]

        result = _compute_hpss(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            hop_length=self._hop_length,
            n_fft=self._n_fft,
            margin=margin,
            audio=audio
        )
        harmonic, percussive = result.value

        self._logger.debug(
            f"Percussive-only extracted: shape={percussive.shape}",
            separator=self._separator
        )
        return result

    def extract_harmonic(
            self,
            *,
            file_path: Path = None,
            audio: np.ndarray = None,
            sample_rate: int,
            tempo_bpm: float
    ) -> TimedCacheResult[tuple[np.ndarray, np.ndarray]]:
        """
        Extracts harmonic & percussive components based on tempo.
        If file_path provided, uses cache; else uses in-memory audio.
        """
        processing_start = time.perf_counter()
        if audio is None and file_path is None:
            raise ValueError("Either `audio` or `file_path` must be provided.")

        # Determine segment length
        length = audio.shape[0] if audio is not None else None
        start_sample = 0
        end_sample = length

        # Compute kernel size based on tempo
        frames_per_beat = sample_rate * 60 / tempo_bpm / self._hop_length
        kernel_size = int(round(frames_per_beat))

        self._logger.debug(
            f"Extracting HPSS with kernel_size={kernel_size}, hop_length={self._hop_length}, n_fft={self._n_fft}",
            separator=self._separator
        )

        processing_duration = time.perf_counter() - processing_start

        results = _compute_hpss(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            hop_length=self._hop_length,
            n_fft=self._n_fft,
            kernel_size=kernel_size,
            audio=audio
        )

        harmonic, percussive = results.value

        self._logger.debug(
            f"Harmonic shape={harmonic.shape}, Percussive shape={percussive.shape}",
            separator=self._separator
        )
        return TimedCacheResult(
            value=(harmonic, percussive),
            time_waiting=results.time_waiting,
            time_processing=processing_duration+results.time_processing,
            retrieved_from_cache=results.retrieved_from_cache,
        )
