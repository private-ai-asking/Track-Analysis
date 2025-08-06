from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.testing import TestInterface
from track_analysis.components.md_common_python.py_common.user_input.user_input_helper import UserInputHelper
from track_analysis.components.track_analysis.legacy.audio_file_handler import AudioFileHandler, AudioStreamsInfoModel


class ShortTimeRMSTest(TestInterface):
    def __init__(self, logger: HoornLogger, audio_file_handler: AudioFileHandler):
        super().__init__(logger, is_child=True)
        self._separator = "ShortTimeRMSTest"
        self._audio_file_handler = audio_file_handler
        self._input_helper: UserInputHelper = UserInputHelper(logger, self._separator)
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def test(self) -> None:
        def _validate_path(path: str) -> Tuple[bool, str]:
            try:
                p: Path = Path(path)
                if not p.exists() or not p.is_file():
                    return False, "Something went wrong when validating the path!"
                return True, ""
            except Exception as e:
                return False, str(e)


        audio_file_path = self._input_helper.get_user_input("Enter the audio file path.", expected_response_type=str, validator_func=_validate_path)
        audio_file_path = Path(audio_file_path)
        info: List[AudioStreamsInfoModel] = self._audio_file_handler.get_audio_streams_info_batch([audio_file_path])
        samples = info[0].samples
        rms = self.compute_short_time_rms(samples)

        mean_rms = rms.mean()
        max_rms = rms.max()
        perc90_rms = np.percentile(rms, 90)

        self._logger.info(f"RMS[mean={mean_rms};max={max_rms};90th={perc90_rms}]")

    @staticmethod
    def compute_short_time_rms(
            samples: np.ndarray,
            frame_length: int = 2048,
            hop_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute short‑time RMS energy for an audio buffer in one vectorized pass.

        Args:
          samples       : np.ndarray, shape=(n_samples,) or (n_samples, n_channels)
          frame_length  : window size in samples (e.g. 2048 for ~46 ms@44.1 kHz)
          hop_length    : step size in samples. If None, defaults to frame_length (no overlap).

        Returns:
          rms           : np.ndarray, shape=(n_frames,), short‑time RMS per window
        """
        # 1) make mono by averaging channels
        if samples.ndim == 2:
            # axis=1 shape: (n_samples, n_channels)
            mono = samples.mean(axis=1)
        else:
            mono = samples

        if hop_length is None:
            hop_length = frame_length

        # 2) trim to an integer number of hops
        total = len(mono)
        n_frames = 1 + (total - frame_length) // hop_length
        trimmed = mono[: hop_length * n_frames + frame_length - hop_length]

        # 3) use a sliding window view to get an (n_frames, frame_length) array
        windows = sliding_window_view(trimmed, frame_length)[::hop_length]

        # 4) RMS = sqrt(mean(square))  — one vectorized call over the whole array
        rms = np.sqrt(np.mean(windows**2, axis=1))

        return rms

