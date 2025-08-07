from pathlib import Path
import numpy as np
import librosa
from librosa.beat import tempo
from librosa.onset import onset_detect

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.cache(identifier_arg="file_path", ignore=["audio"])
def compute_onset_strengths(
        *,
        file_path:     Path,
        start_sample:  int,
        end_sample:    int,
        sample_rate:   int,
        hop_length:    int,
        unique_string: str,
        audio:         np.ndarray = None,
) -> np.ndarray:
    """
    Returns the onset-strength envelope for the given range.
    Caches on (file_path, start_sample, end_sample, sample_rate, hop_length) only.

    If `audio` is provided, slices that array. Otherwise memory-maps the file.
    """
    if audio is None:
        audio = np.memmap(
            str(file_path),
            dtype="float32",
            mode="r",
            offset=start_sample * 4,
            shape=(end_sample - start_sample,),
        )
    else:
        audio = audio[start_sample:end_sample]

    return librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length
    )


@MEMORY.cache(identifier_arg="file_path", ignore=["audio"])
def compute_onset_peaks(
        *,
        file_path:     Path,
        start_sample:  int,
        end_sample:    int,
        sample_rate:   int,
        hop_length:    int,
        unique_string: str,
        audio:         np.ndarray = None,
) -> np.ndarray:
    """
    Returns the frame indices of detected onset peaks for the given range.
    Caches on same key as compute_onset_strengths.
    """
    # reuse cached onset-strength envelope
    env = compute_onset_strengths(
        file_path=file_path,
        start_sample=start_sample,
        end_sample=end_sample,
        sample_rate=sample_rate,
        hop_length=hop_length,
        audio=audio,
        unique_string=unique_string,
    )
    # detect peaks
    peaks = onset_detect(
        onset_envelope=env,
        sr=sample_rate,
        hop_length=hop_length
    )
    return peaks


@MEMORY.cache(identifier_arg="file_path", ignore=["audio"])
def compute_dynamic_tempo(
        *,
        file_path: Path,
        start_sample: int,
        end_sample: int,
        sample_rate: int,
        hop_length: int,
        audio: np.ndarray = None,
) -> np.ndarray:
    """
    Returns the dynamic tempo for the given range.
    Caches on same key as compute_onset_strengths.
    """
    # reuse cached onset-strength envelope
    env = compute_onset_strengths(
        file_path=file_path,
        start_sample=start_sample,
        end_sample=end_sample,
        sample_rate=sample_rate,
        hop_length=hop_length,
        audio=audio,
    )
    return tempo(
        onset_envelope=env,
        sr=sample_rate,
        hop_length=hop_length,
        aggregate=None
    )


class OnsetStrengthExtractor:
    def __init__(self, logger: HoornLogger):
        self._logger    = logger
        self._separator = self.__class__.__name__

    def extract_envelope(
            self,
            file_path:     Path,
            start_sample:  int,
            end_sample:    int,
            sample_rate:   int,
            hop_length:    int,
            unique_string: str,
            audio:         np.ndarray = None,
    ) -> np.ndarray:
        """
        Returns the onset-strength envelope for the given range.
        """
        self._logger.debug(
            f"Extracting onset strength for {file_path.name}[{start_sample}:{end_sample}]",
            separator=self._separator,
        )
        return compute_onset_strengths(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            hop_length=hop_length,
            audio=audio,
            unique_string=unique_string,
        )

    def extract_peaks(
            self,
            file_path:     Path,
            start_sample:  int,
            end_sample:    int,
            sample_rate:   int,
            hop_length:    int,
            unique_string: str,
            audio:         np.ndarray = None,
    ) -> np.ndarray:
        """
        Returns the onset peak frame indices for the given range.
        """
        self._logger.debug(
            f"Extracting onset peaks for {file_path.name}[{start_sample}:{end_sample}]",
            separator=self._separator,
        )
        return compute_onset_peaks(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            hop_length=hop_length,
            unique_string=unique_string,
            audio=audio,
        )

    def extract_dynamic_tempo(
            self,
            file_path: Path,
            start_sample: int,
            end_sample: int,
            sample_rate: int,
            hop_length: int,
            audio: np.ndarray = None,
    ) -> np.ndarray:
        """
        Returns the dynamic tempo for the given range.
        """
        self._logger.debug(
            f"Extracting dynamic tempo for {file_path.name}[{start_sample}:{end_sample}]",
            separator=self._separator,
        )
        return compute_dynamic_tempo(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            hop_length=hop_length,
            audio=audio,
        )
