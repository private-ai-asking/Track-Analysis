from pathlib import Path
import numpy as np
import librosa

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.core.cacheing.shared import MEMORY


@MEMORY.cache(ignore=["audio"])
def compute_onset_strengths(
        *,
        file_path:     Path,
        start_sample:  int,
        end_sample:    int,
        sample_rate:   int,
        hop_length:    int,
        audio:         np.ndarray = None,
) -> np.ndarray:
    """
    Returns the onset-strength envelope for the given range.
    Caches on (file_path, start_sample, end_sample, sample_rate, hop_length) only.

    If `audio` is provided, slices that array. Otherwise memory-maps the file.
    """
    if audio is None:
        # float32 = 4 bytes/sample
        audio = np.memmap(
            str(file_path),
            dtype="float32",
            mode="r",
            offset=start_sample * 4,
            shape=(end_sample - start_sample,),
        )
    else:
        # use the in-memory portion
        audio = audio[start_sample:end_sample]

    return librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length
    )


class OnsetStrengthExtractor:
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
            audio:         np.ndarray = None,
    ) -> np.ndarray:
        """
        Always gives you a cached result keyed by file & indices.
        Pass `audio` if you’ve already decoded it, for in-memory slicing;
        otherwise it will memmap from disk.
        """
        self._logger.debug(
            f"Extracting onsets for {file_path.name}[{start_sample}:{end_sample}]",
            separator=self._separator,
        )
        return compute_onset_strengths(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            hop_length=hop_length,
            audio=audio,
        )
