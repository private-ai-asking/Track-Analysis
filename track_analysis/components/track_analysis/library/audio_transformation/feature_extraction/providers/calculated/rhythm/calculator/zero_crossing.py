from pathlib import Path

import numpy as np
from librosa.feature import zero_crossing_rate

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.shared.caching.hdf5_memory import TimedCacheResult
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.timed_cache(identifier_arg="file_path", ignore=["audio"])
def compute_zero_crossing_rate(
        *,
        file_path:     Path,
        start_sample:  int,
        end_sample:    int,
        sample_rate:   int,
        hop_length:    int,
        audio:         np.ndarray = None,
) -> TimedCacheResult[np.ndarray]:
    """
    Returns the zero-crossing rate for the given range.
    """
    audio = audio[start_sample:end_sample]

    # noinspection PyTypeChecker
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
    ) -> TimedCacheResult[np.ndarray]:
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
