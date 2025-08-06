from pathlib import Path
import numpy as np
from librosa.feature import spectral_rolloff

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.cache(identifier_arg="file_path", ignore=["audio"])
def compute_spectral_rolloff(
        *,
        file_path:     Path,
        start_sample:  int,
        end_sample:    int,
        sample_rate:   int,
        hop_length:    int,
        roll_percent:  float = 0.85,
        audio:         np.ndarray = None,
) -> np.ndarray:
    """
    Returns the spectral rolloff for the given range.
    Caches on (file_path, start_sample, end_sample, sample_rate, hop_length, roll_percent) only.

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

    return spectral_rolloff(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        roll_percent=roll_percent,
    )


class SpectralRolloffExtractor:
    def __init__(self, logger: HoornLogger):
        self._logger    = logger
        self._separator = self.__class__.__name__

    def extract(
            self,
            file_path:     Path,
            start_sample:  int,
            end_sample:    int,
            sample_rate:   int,
            hop_length:    int,
            roll_percent:  float = 0.85,
            audio:         np.ndarray = None,
    ) -> np.ndarray:
        """
        Returns the spectral rolloff for the given range.
        """
        self._logger.debug(
            f"Extracting spectral rolloff for {file_path.name}[{start_sample}:{end_sample}]",
            separator=self._separator,
        )
        return compute_spectral_rolloff(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            hop_length=hop_length,
            roll_percent=roll_percent,
            audio=audio,
        )
