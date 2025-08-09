from pathlib import Path

import numpy as np
from librosa.feature import mfcc, delta

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.shared.caching.hdf5_memory import TimedCacheResult
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.timed_cache(identifier_arg="file_path", ignore=["audio"])
def _compute_mfccs(
        *,
        file_path:     Path,
        start_sample:  int,
        end_sample:    int,
        sample_rate:   int,
        n_mfcc:        int = 20,
        audio:         np.ndarray = None,
) -> TimedCacheResult[np.ndarray]:
    """
    Returns the Mel-frequency cepstral coefficients (MFCCs) for the given range.
    Caches on (file_path, start_sample, end_sample, sample_rate, n_mfcc) only.
    """
    audio = audio[start_sample:end_sample]

    # noinspection PyTypeChecker
    return mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
    )

@MEMORY.timed_cache(identifier_arg="file_path",
              ignore=["mffccs", "sample_rate"])
def _compute_deltas(
        *,
        file_path: Path,
        start_sample: int,
        end_sample: int,
        sample_rate: int,
        order: int = 1,
        axis: int = -1,
        mffccs: np.ndarray,
) -> TimedCacheResult[np.ndarray]:
    segment = mffccs[start_sample:end_sample]

    width = min(9, segment.shape[axis] if axis >= 0 else segment.shape[segment.ndim + axis])

    # noinspection PyTypeChecker
    return delta(
        data=segment,
        width=width,
        order=order,
        axis=axis,
    )


class MfccExtractor:
    def __init__(self, logger: HoornLogger):
        self._logger    = logger
        self._separator = self.__class__.__name__

    def extract_mfccs(
            self,
            file_path:     Path,
            start_sample:  int,
            end_sample:    int,
            sample_rate:   int,
            n_mfcc:        int = 20,
            audio:         np.ndarray = None,
    ) -> TimedCacheResult[np.ndarray]:
        """
        Returns the MFCCs for the given range.
        """
        self._logger.debug(
            f"Extracting MFCCs for {file_path.name}[{start_sample}:{end_sample}]",
            separator=self._separator,
        )
        return _compute_mfccs(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            audio=audio,
        )

    def extract_deltas(self,
                       *,
                       file_path:     Path,
                       start_sample:  int,
                       end_sample:    int,
                       sample_rate:   int,
                       order: int = 1,
                       axis: int = -1,
                       mffccs:         np.ndarray,) -> TimedCacheResult[np.ndarray]:
        self._logger.debug(
            f"Extracting Deltas for {file_path.name}[{start_sample}:{end_sample}]",
            separator=self._separator,
        )

        return _compute_deltas(
            file_path=file_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            order=order,
            axis=axis,
            mffccs=mffccs,
        )

