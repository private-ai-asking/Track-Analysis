from pathlib import Path

import numpy as np
from librosa.feature import zero_crossing_rate

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.core.cacheing.shared import MEMORY


@MEMORY.cache(ignore=["audio"])
def compute_zero_crossing_rate(
        *,
        file_path:     Path,
        start_sample:  int,
        end_sample:    int,
        sample_rate:   int,
        hop_length:    int,
        audio:         np.ndarray = None,
) -> np.ndarray:
    """
    Returns the zero-crossing rate for the given range.
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

    return zero_crossing_rate(
        y=audio,
        hop_length=hop_length
    )


class ZeroCrossingRateExtractor:
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
        Extracts zero-crossing rate envelope, cached.
        """
        self._logger.debug(
            f"ZeroCrossingRateExtractor: {file_path.name}[{start_sample}:{end_sample}]",
            separator=self._separator,
        )
        return compute_zero_crossing_rate(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            hop_length=hop_length,
            audio=audio,
        )
