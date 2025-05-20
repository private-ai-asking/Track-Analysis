from pathlib import Path

import librosa
import pydantic
from numpy import ndarray

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
    samples_librosa: ndarray

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

    def get_audio_streams_info(self, audio_file: Path) -> AudioStreamsInfoModel:
        """
        Retrieves audio stream information from a file.

        Args:
            audio_file: Path to the audio file.

        Returns:
            AudioStreamsInfoModel: An object containing audio stream information.  Returns an object with default values in case of error.
        """
        try:
            info = self._ffprobe_client.run_ffprobe(str(audio_file))
            return self._extract_audio_info(info, audio_file)

        except FileNotFoundError:
            self._logger.error("ffprobe not found. Please install FFmpeg.", separator=self._separator)
            return AudioStreamsInfoModel(duration=0.0, bitrate=0, sample_rate_kHz=0, sample_rate_Hz=0, bit_depth=0, channels=0, format="")
        except FFprobeError as e:
            self._logger.warning(f"ffprobe error: {e}", separator=self._separator)
            return AudioStreamsInfoModel(duration=0.0, bitrate=0, sample_rate_kHz=0, sample_rate_Hz=0, bit_depth=0, channels=0, format="")
        except Exception as e:
            self._logger.warning(f"An unexpected error occurred: {e}", separator=self._separator)
            return AudioStreamsInfoModel(duration=0.0, bitrate=0, sample_rate_kHz=0, sample_rate_Hz=0, bit_depth=0, channels=0, format="")

    def _extract_audio_info(self, info: dict, audio_file: Path) -> AudioStreamsInfoModel:
        """Extracts audio information from ffprobe output."""
        duration = self._extract_duration(info)
        bitrate = self._extract_bitrate(info)
        sample_rate, bit_depth, audio_format, channels = self._extract_stream_info(info)

        samples_librosa, sr = librosa.load(audio_file, sr=None)

        if sr != sample_rate:
            self._logger.warning(f"Librosa reported a different sample rate than ffprobe for: {audio_file}, {sr} vs {sample_rate}.", separator=self._separator)

        return AudioStreamsInfoModel(duration=duration, bitrate=bitrate / 1000, sample_rate_kHz=sample_rate / 1000, sample_rate_Hz=sample_rate,
                                     bit_depth=bit_depth, channels=channels, format=audio_format, samples_librosa=samples_librosa)

    def _extract_duration(self, info: dict) -> float:
        """Extracts duration from the ffprobe 'format' section."""
        duration = 0.0
        if 'format' in info and 'duration' in info['format']:
            try:
                duration = float(info['format']['duration'])
            except ValueError:
                self._logger.warning(
                    f"Could not convert duration to float. Raw duration: {info['format']['duration']}",
                    separator=self._separator)
        return duration

    def _extract_bitrate(self, info: dict) -> int:
        """Extracts bitrate from the ffprobe 'format' section."""
        bitrate = 0
        if 'format' in info and 'bit_rate' in info['format']:
            try:
                bitrate = int(info['format']['bit_rate'])
            except ValueError:
                self._logger.warning(
                    f"Could not convert bitrate to int. Raw bitrate: {info['format']['bit_rate']}",
                    separator=self._separator)
        return bitrate

    def _extract_stream_info(self, info: dict) -> tuple[int, int, str, int]:
        """Extracts stream-specific information from the ffprobe output."""
        sample_rate = 0
        bit_depth = 0
        audio_format = ""
        channels = 0

        if 'streams' in info:
            for stream in info['streams']:
                if 'sample_rate' in stream:
                    sample_rate = self._parse_int(stream['sample_rate'], "sample_rate")
                if 'bits_per_sample' in stream:
                    bit_depth = self._parse_int(stream['bits_per_sample'], "bit_depth")
                if 'sample_fmt' in stream and bit_depth == 0:
                    audio_format = stream["sample_fmt"]
                    bit_depth = self._audio_format_converter.get_bit_depth_from_format(audio_format)
                if 'channels' in stream:
                    channels = self._parse_int(stream['channels'], "channels")

        return sample_rate, bit_depth, audio_format, channels

    def _parse_int(self, value: str, field_name: str) -> int:
        """Helper function to parse an integer value from the ffprobe output."""
        try:
            return int(value)
        except ValueError:
            self._logger.warning(f"Could not convert {field_name} to int. Raw {field_name}: {value}",
                                 separator=self._separator)
            return 0
