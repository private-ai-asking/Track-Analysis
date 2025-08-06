from pathlib import Path

import numpy as np
from librosa.feature import mfcc

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.shared_objects import MEMORY


@MEMORY.cache(identifier_arg="file_path", ignore=["audio"])
def _compute_mfccs(
        *,
        file_path:     Path,
        start_sample:  int,
        end_sample:    int,
        sample_rate:   int,
        n_mfcc:        int = 20,
        audio:         np.ndarray = None,
) -> np.ndarray:
    """
    Returns the Mel-frequency cepstral coefficients (MFCCs) for the given range.
    Caches on (file_path, start_sample, end_sample, sample_rate, n_mfcc) only.
    """
    audio = audio[start_sample:end_sample]

    return mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
    )


class MfccExtractor:
    def __init__(self, logger: HoornLogger):
        self._logger    = logger
        self._separator = self.__class__.__name__

    def extract(
            self,
            file_path:     Path,
            start_sample:  int,
            end_sample:    int,
            sample_rate:   int,
            n_mfcc:        int = 20,
            audio:         np.ndarray = None,
    ) -> np.ndarray:
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
