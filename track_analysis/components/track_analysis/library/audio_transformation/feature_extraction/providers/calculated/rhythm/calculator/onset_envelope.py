from pathlib import Path
import numpy as np
import librosa
from librosa.beat import tempo
from librosa.onset import onset_detect

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.shared.caching.hdf5_memory import TimedCacheResult
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.timed_cache(identifier_arg="file_path", ignore=["audio"])
def compute_onset_strengths(
        *,
        file_path:     Path,
        start_sample:  int,
        end_sample:    int,
        sample_rate:   int,
        hop_length:    int,
        unique_string: str,
        audio:         np.ndarray = None,
) -> TimedCacheResult[np.ndarray]:
    audio = audio[start_sample:end_sample]

    # noinspection PyTypeChecker
    return librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length
    )


@MEMORY.timed_cache(identifier_arg="file_path", ignore=["onset_envelope"])
def compute_onset_peaks(
        *,
        file_path:     Path,
        start_sample:  int,
        end_sample:    int,
        sample_rate:   int,
        hop_length:    int,
        unique_string: str,
        onset_envelope: np.ndarray,
) -> TimedCacheResult[np.ndarray]:
    # detect peaks
    peaks = onset_detect(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop_length
    )
    return peaks # type: ignore


@MEMORY.timed_cache(identifier_arg="file_path", ignore=["onset_envelope"])
def compute_dynamic_tempo(
        *,
        file_path: Path,
        start_sample: int,
        end_sample: int,
        sample_rate: int,
        hop_length: int,
        onset_envelope: np.ndarray,
) -> TimedCacheResult[np.ndarray]:
    return tempo(
        onset_envelope=onset_envelope,
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
    ) -> TimedCacheResult[np.ndarray]:
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
    ) -> TimedCacheResult[np.ndarray]:
        """
        Returns the onset peak frame indices for the given range.
        """
        self._logger.debug(
            f"Extracting onset peaks for {file_path.name}[{start_sample}:{end_sample}]",
            separator=self._separator,
        )
        onset_env_results = self.extract_envelope(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            hop_length=hop_length,
            unique_string=unique_string,
            audio=audio,
        )

        peak_results = compute_onset_peaks(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            hop_length=hop_length,
            unique_string=unique_string,
            onset_envelope=onset_env_results.value,
        )

        return TimedCacheResult(
            value=peak_results.value,
            time_waiting=onset_env_results.time_waiting+peak_results.time_waiting,
            time_processing=onset_env_results.time_processing+peak_results.time_processing,
            retrieved_from_cache=peak_results.retrieved_from_cache,
        )

    def extract_dynamic_tempo(
            self,
            file_path: Path,
            start_sample: int,
            end_sample: int,
            sample_rate: int,
            hop_length: int,
            audio: np.ndarray = None,
    ) -> TimedCacheResult[np.ndarray]:
        """
        Returns the dynamic tempo for the given range.
        """
        self._logger.debug(
            f"Extracting dynamic tempo for {file_path.name}[{start_sample}:{end_sample}]",
            separator=self._separator,
        )

        onset_env_results = self.extract_envelope(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            hop_length=hop_length,
            unique_string="dynamic-tempo",
            audio=audio,
        )

        dynamic_tempo_results = compute_dynamic_tempo(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            hop_length=hop_length,
            onset_envelope=onset_env_results.value,
        )

        return TimedCacheResult(
            value=dynamic_tempo_results.value,
            time_waiting=onset_env_results.time_waiting+dynamic_tempo_results.time_waiting,
            time_processing=onset_env_results.time_processing+dynamic_tempo_results.time_processing,
            retrieved_from_cache=dynamic_tempo_results.retrieved_from_cache,
        )
