from pathlib import Path

import librosa
import numpy as np
import pydantic
from mutagen import flac
from mutagen.flac import FLAC

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


class AddAdvancedMetadata(IPipe):
    def __init__(self, logger: HoornLogger):
        self._separator = "AddAdvancedMetadataPipe"

        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def _get_stream_info(self, audio_file: Path) -> StreamInfoModel:
        if audio_file.suffix == ".flac":
            flac_file = FLAC(audio_file)
            stream_info_raw: flac.StreamInfo = flac_file.info
            return StreamInfoModel(
                duration=round(stream_info_raw.length, 4),
                bitrate=stream_info_raw.bitrate / 1000,
                sample_rate=stream_info_raw.sample_rate / 1000
            )
        else:
            self._logger.warning(f"Unable to read stream info from audio: {audio_file}", separator=self._separator)
            return StreamInfoModel(duration=0, bitrate=0, sample_rate=0)

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

    def _get_bit_depth(self, audio_file: Path):
        info = soundfile.info(audio_file)
        if info.subtype == 'PCM_16':
            bit_depth = 16
        elif info.subtype == 'PCM_24':
            bit_depth = 24
        elif info.subtype == 'PCM_32':
            bit_depth = 32
        elif info.subtype == 'FLOAT':
            bit_depth = 32
        elif info.subtype == 'DOUBLE':
            bit_depth = 64
        else:
            bit_depth = 0
            self._logger.warning(f"Unhandled subtype: {info.subtype} for track: {audio_file}", separator=self._separator)

        return bit_depth

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
            track.metadata.append(AudioMetadataItem(header=Header.Bit_Depth, description="The bit depth of the track in bits.", value=self._get_bit_depth(track.path)))
            self._logger.trace(f"Finished adding metadata for track: {track.path}", separator=self._separator)

        self._logger.trace("Finished adding advanced metadata.", separator=self._separator)
        return data
