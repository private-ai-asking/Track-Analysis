import json
import subprocess
from pathlib import Path

import pydantic

from track_analysis.components.md_common_python.py_common.logging import HoornLogger

class AudioStreamsInfoModel(pydantic.BaseModel):
    duration: float
    bitrate: float
    sample_rate: float
    bit_depth: int
    channels: int
    format: str


class AudioFileHandler:
    def __init__(self, logger: HoornLogger):
        self._separator = "AudioFileHandler"
        self._logger = logger

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def _get_bit_depth_from_format(self, sample_format: str) -> int:
        """
        Gets the bit depth from the FFmpeg sample format string.

        Args:
            sample_format: The FFmpeg sample format string (e.g., "s16le", "flt", "dbl").

        Returns:
            The bit depth as an integer (e.g., 16, 32, 64), or None if the format is unknown or invalid.
        """
        format_lower = sample_format.lower()

        if format_lower.startswith("s") or format_lower.startswith("u"):
            try:
                bits = int(format_lower[1:3])
                return bits
            except ValueError:
                pass
        elif format_lower.startswith("flt") or format_lower.startswith("s32p"):
            return 32
        elif format_lower.startswith("dbl"):
            return 64
        elif format_lower.startswith("s16p"):
            return 16
        elif format_lower.startswith("u8") or format_lower.startswith("s8"):
            return 8

        self._logger.warning(f"Unsupported format for bit depth extraction: {format_lower}", separator=self._separator)
        return 0

    def get_audio_streams_info(self, audio_file: Path) -> AudioStreamsInfoModel:
        try:
            audio_file_path = str(audio_file)

            cmd = ["ffprobe", '-v', 'error', '-show_format', '-show_streams', '-of', 'json', audio_file_path]
            self._logger.debug(f"Running command: {' '.join(cmd)}", separator=self._separator)
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")

            if result.returncode == 0:
                try:
                    info = json.loads(result.stdout)
                except json.JSONDecodeError as e:
                    self._logger.warning(f"Error decoding JSON from ffprobe output: {e}", separator=self._separator)
                    self._logger.warning(f"ffprobe output: {result.stdout}", separator=self._separator)
                    return AudioStreamsInfoModel(duration=0.0, bitrate=0, sample_rate=0, bit_depth=0, channels=0, format="")

                duration = 0.0
                bitrate = 0
                sample_rate = 0
                bit_depth = 0
                channels = 0
                audio_format = ""

                if 'format' in info:
                    if 'duration' in info['format']:
                        try:
                            duration = float(info['format']['duration'])
                        except ValueError:
                            self._logger.warning(f"Could not convert duration to float. Raw duration: {info['format']['duration']}", separator=self._separator)
                    if 'bit_rate' in info['format']:
                        try:
                            bitrate = int(info['format']['bit_rate'])
                        except ValueError:
                            self._logger.warning(f"Could not convert bitrate to int. Raw bitrate: {info['format']['bit_rate']}", separator=self._separator)

                if 'streams' in info:
                    for stream in info['streams']:
                        if 'sample_rate' in stream:
                            try:
                                sample_rate = int(stream['sample_rate'])
                            except ValueError:
                                self._logger.warning(f"Could not convert sample_rate to int. Raw sample_rate: {stream['sample_rate']}", separator=self._separator)
                        if 'bits_per_sample' in stream:
                            try:
                                bit_depth = int(stream['bits_per_sample'])
                            except ValueError:
                                self._logger.warning(f"Could not convert bit_depth to int. Raw bit_depth: {stream['bits_per_sample']}", separator=self._separator)
                        if 'sample_fmt' in stream and bit_depth == 0:
                            bit_depth = self._get_bit_depth_from_format(stream['sample_fmt'])
                            audio_format = stream["sample_fmt"]
                        if 'channels' in stream:
                            try:
                                channels = int(stream['channels'])
                            except ValueError:
                                self._logger.warning(f"Could not convert channels to int. Raw channels: {stream['channels']}", separator=self._separator)

                return AudioStreamsInfoModel(duration=duration, bitrate=bitrate / 1000, sample_rate=sample_rate / 1000, bit_depth=bit_depth, channels=channels, format=audio_format)

            else:
                self._logger.warning(f"ffprobe error: {result.stderr}", separator=self._separator)
                return AudioStreamsInfoModel(duration=0.0, bitrate=0, sample_rate=0, bit_depth=0, channels=0, format="")

        except FileNotFoundError:
            self._logger.error("ffprobe not found. Please install FFmpeg.", separator=self._separator)
            return AudioStreamsInfoModel(duration=0.0, bitrate=0, sample_rate=0, bit_depth=0, channels=0, format="")
        except Exception as e:
            self._logger.warning(f"An unexpected error occurred: {e}", separator=self._separator)
            return AudioStreamsInfoModel(duration=0.0, bitrate=0, sample_rate=0, bit_depth=0, channels=0, format="")
