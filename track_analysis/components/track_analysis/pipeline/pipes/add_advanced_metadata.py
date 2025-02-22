import json
import subprocess
from pathlib import Path

import librosa
import numpy as np
import pydantic

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.model.audio_metadata_item import AudioMetadataItem
from track_analysis.components.track_analysis.model.header import Header
from track_analysis.components.track_analysis.pipeline.pipeline_context import PipelineContextModel

import soundfile

class StreamInfoModel(pydantic.BaseModel):
    duration: float
    bitrate: float
    sample_rate: float
    bit_depth: int


class AddAdvancedMetadata(IPipe):
    def __init__(self, logger: HoornLogger):
        self._separator = "AddAdvancedMetadataPipe"

        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

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

        elif format_lower.startswith("flt"):
            return 32
        elif format_lower.startswith("dbl"):
            return 64
        elif format_lower.startswith("s32p"):
            return 32
        elif format_lower.startswith("s16p"):
            return 16
        elif format_lower.startswith("u8"):
            return 8
        elif format_lower.startswith("s8"):
            return 8

        # Add more formats as needed (e.g., planar formats, other integer sizes)
        # Be sure to handle potential errors (e.g., invalid format strings).

        self._logger.warning(f"Unsupported format for bit depth extraction: {format_lower}", separator=self._separator)
        return 0

    def _get_stream_info(self, audio_file: Path) -> StreamInfoModel:
        try:
            audio_file_path = str(audio_file)

            cmd = ["ffprobe", '-v', 'error', '-show_format', '-show_streams', '-of', 'json', audio_file_path]
            self._logger.debug(f"Running command: {' '.join(cmd)}", separator=self._separator)
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if result.returncode == 0:
                try:
                    info = json.loads(result.stdout)
                except json.JSONDecodeError as e:
                    self._logger.warning(f"Error decoding JSON from ffprobe output: {e}", separator=self._separator)
                    self._logger.warning(f"ffprobe output: {result.stdout}", separator=self._separator)
                    return StreamInfoModel(duration=0.0, bitrate=0, sample_rate=0, bit_depth=0)

                duration = 0.0
                bitrate = 0
                sample_rate = 0
                bit_depth = 0

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

                return StreamInfoModel(duration=duration, bitrate=bitrate/1000, sample_rate=sample_rate/1000, bit_depth=bit_depth)

            else:
                self._logger.warning(f"ffprobe error: {result.stderr}", separator=self._separator)
                return StreamInfoModel(duration=0.0, bitrate=0, sample_rate=0, bit_depth=0)

        except FileNotFoundError:
            self._logger.error("ffprobe not found. Please install FFmpeg.", separator=self._separator)
            return StreamInfoModel(duration=0.0, bitrate=0, sample_rate=0, bit_depth=0)
        except Exception as e:
            self._logger.warning(f"An unexpected error occurred: {e}", separator=self._separator)
            return StreamInfoModel(duration=0.0, bitrate=0, sample_rate=0, bit_depth=0)

    def _calculate_dynamic_range_and_crest_factor(self, audio_file: Path) -> (float, float):
        """Calculates the peak-to-RMS dynamic range of an audio signal.

         Args:
             audio_file (str): Path to the audio file.

         Returns:
             float: The dynamic range in dB, or None if an error occurs.
         """
        try:
            y, sr = librosa.load(audio_file)  # Load the audio
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return 0, 0

        peak_amplitude = np.max(np.abs(y))
        rms_amplitude = np.sqrt(np.mean(y**2))

        if rms_amplitude == 0:
            return float('inf') if peak_amplitude > 0 else -float('inf')
        dynamic_range = 20 * np.log10(peak_amplitude / rms_amplitude)
        crest_factor = peak_amplitude / rms_amplitude

        return dynamic_range, crest_factor

    def flow(self, data: PipelineContextModel) -> PipelineContextModel:
        self._logger.trace("Adding advanced metadata...", separator=self._separator)

        for track in data.audio_info:
            self._logger.trace(f"Adding metadata for track: {track.path}...", separator=self._separator)
            file_info: StreamInfoModel = self._get_stream_info(track.path)
            dynamic_range, crest_factor = self._calculate_dynamic_range_and_crest_factor(track.path)

            track.metadata.append(AudioMetadataItem(header=Header.Duration, description="The duration of the track in seconds.", value=file_info.duration))
            track.metadata.append(AudioMetadataItem(header=Header.Bitrate, description="The bitrate of the track in kbps.", value=file_info.bitrate))
            track.metadata.append(AudioMetadataItem(header=Header.Sample_Rate, description="The sample rate of the track in Hz.", value=file_info.sample_rate))
            track.metadata.append(AudioMetadataItem(header=Header.Peak_To_RMS, description="The peak-to-RMS dynamic range of the track in dB.", value=dynamic_range))
            track.metadata.append(AudioMetadataItem(header=Header.Crest_Factor, description="The crest factor of the track.", value=crest_factor))
            track.metadata.append(AudioMetadataItem(header=Header.Bit_Depth, description="The bit depth of the track in bits.", value=file_info.bit_depth))
            self._logger.trace(f"Finished adding metadata for track: {track.path}", separator=self._separator)

        self._logger.trace("Finished adding advanced metadata.", separator=self._separator)
        return data
