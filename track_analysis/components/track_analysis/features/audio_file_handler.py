from pathlib import Path
from typing import Optional, List

import soundfile as sf
import pydantic
from numpy import ndarray
from pymediainfo import MediaInfo

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.apis.ffprobe_client import FFprobeClient
from track_analysis.components.track_analysis.util.audio_format_converter import AudioFormatConverter


class AudioStreamsInfoModel(pydantic.BaseModel):
    duration: float
    bitrate: float
    sample_rate_kHz: float
    sample_rate_Hz: float
    bit_depth: int
    channels: int
    format: str
    samples: Optional[ndarray] = None

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
        """Extracts audio information from ffprobe and reads samples at full native precision."""
        # --- 1) MediaInfo metadata ---
        media_info   = MediaInfo.parse(str(audio_file))
        audio        = media_info.audio_tracks[0]

        duration_s   = float(audio.duration) / 1000.0
        bitrate_bps  = int(audio.bit_rate)
        sample_rate  = int(audio.sampling_rate)
        bit_depth    = int(audio.bit_depth) if audio.bit_depth else None
        channels     = int(audio.channel_s)
        audio_format = audio.format

        # --- 2) read raw samples in one call, auto-detecting container/codec ---
        samples, sr = sf.read(str(audio_file), dtype="float64", always_2d=True)
        # samples.shape == (frames, channels)

        # --- 3) sanity-check sample rate match ---
        if sr != sample_rate:
            self._logger.warning(
                f"Sample-rate mismatch for {audio_file}: "
                f"media_info={sample_rate} Hz vs soundfile={sr} Hz",
                separator=self._separator
            )

        # --- 4) return structured info (you can adjust field names as needed) ---
        return AudioStreamsInfoModel(
            duration        = duration_s,
            bitrate         = bitrate_bps / 1000,
            sample_rate_kHz = sample_rate / 1000,
            sample_rate_Hz  = sample_rate,
            bit_depth       = bit_depth,
            channels        = channels,
            format          = audio_format,
            samples= samples,
        )
