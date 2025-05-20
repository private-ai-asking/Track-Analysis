from pathlib import Path
from typing import Optional, List, Dict

import librosa
import pydantic
from numpy import ndarray
from pymediainfo import MediaInfo

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.apis.ffprobe_client import FFprobeClient
from track_analysis.components.track_analysis.exceptions.ffprobe_error import FFprobeError
from track_analysis.components.track_analysis.util.audio_format_converter import AudioFormatConverter


class AudioStreamsInfoModel(pydantic.BaseModel):
    duration: float
    bitrate: float
    sample_rate_kHz: float
    sample_rate_Hz: float
    bit_depth: int
    channels: int
    format: str
    samples_librosa: Optional[ndarray] = None

    model_config = {
        "arbitrary_types_allowed": True
    }


class AudioFileHandler:
    """Handles audio file information retrieval using ffprobe."""

    def __init__(self, logger: HoornLogger):
        self._separator = "AudioFileHandler"
        self._logger = logger
        self._ffprobe_client = FFprobeClient(logger)
        self._audio_format_converter = AudioFormatConverter(logger)

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def get_audio_streams_info_batch(self, audio_files: List[Path]) -> List[AudioStreamsInfoModel]:
        models: List[AudioStreamsInfoModel] = []
        to_process: int = len(audio_files)

        for i, audio_file in enumerate(audio_files):
            models.append(self._extract_audio_info(audio_file))
            self._logger.info(f"Processed {i}/{to_process} ({i/to_process*100:.4f}%)", separator=self._separator)

        return models

    def _extract_audio_info(self, audio_file: Path) -> AudioStreamsInfoModel:
        """Extracts audio information from ffprobe output."""
        media_info = MediaInfo.parse(str(audio_file))
        audio = media_info.audio_tracks[0]

        duration_s   = float(audio.duration) / 1000.0
        bitrate_bps  = int(audio.bit_rate)
        sample_rate  = int(audio.sampling_rate)
        bit_depth    = int(audio.bit_depth) if audio.bit_depth else None
        channels     = int(audio.channel_s)
        audio_format = audio.format

        samples_librosa, sr = librosa.load(audio_file, sr=None)

        if sr != sample_rate:
            self._logger.warning(f"Librosa reported a different sample rate than ffprobe for: {audio_file}, {sr} vs {sample_rate}.", separator=self._separator)

        return AudioStreamsInfoModel(duration=duration_s, bitrate=bitrate_bps / 1000, sample_rate_kHz=sample_rate / 1000, sample_rate_Hz=sample_rate,
                                     bit_depth=bit_depth, channels=channels, format=audio_format, samples_librosa=samples_librosa)
