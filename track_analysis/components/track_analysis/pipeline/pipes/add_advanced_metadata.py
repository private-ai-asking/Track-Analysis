import os

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.audio_calculator import AudioCalculator
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler, AudioStreamsInfoModel
from track_analysis.components.track_analysis.model.audio_metadata_item import AudioMetadataItem
from track_analysis.components.track_analysis.model.header import Header
from track_analysis.components.track_analysis.pipeline.pipeline_context import PipelineContextModel


class AddAdvancedMetadata(IPipe):
    def __init__(self, logger: HoornLogger, audio_file_handler: AudioFileHandler, audio_calculator: AudioCalculator):
        self._separator = "AddAdvancedMetadataPipe"

        self._logger = logger
        self._audio_file_handler = audio_file_handler
        self._audio_calculator = audio_calculator
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: PipelineContextModel) -> PipelineContextModel:
        self._logger.trace("Adding advanced metadata...", separator=self._separator)

        for track in data.generated_audio_info:
            self._logger.trace(f"Adding metadata for track: {track.path}...", separator=self._separator)
            file_info: AudioStreamsInfoModel = self._audio_file_handler.get_audio_streams_info(track.path)
            dynamic_range, crest_factor = self._audio_calculator.calculate_dynamic_range_and_crest_factor(file_info.samples_librosa)
            max_data_per_second = self._audio_calculator.calculate_max_data_per_second(file_info)
            lufs = self._audio_calculator.calculate_lufs(file_info.sample_rate_Hz, file_info.samples_librosa)

            file_size_bytes = os.path.getsize(track.path)
            file_size_bits = file_size_bytes * 8  # Convert bytes to bits
            actual_data_rate = file_size_bits / file_info.duration if file_info.duration > 0 else 0

            efficiency = (actual_data_rate / max_data_per_second) * 100 if max_data_per_second > 0 else 0

            track.metadata.append(AudioMetadataItem(header=Header.Duration, description="The duration of the track in seconds.", value=file_info.duration))
            track.metadata.append(AudioMetadataItem(header=Header.Bitrate, description="The bitrate of the track in kbps.", value=file_info.bitrate))
            track.metadata.append(AudioMetadataItem(header=Header.Sample_Rate, description="The sample rate of the track in kHz.", value=file_info.sample_rate_kHz))
            track.metadata.append(AudioMetadataItem(header=Header.Peak_To_RMS, description="The peak-to-RMS dynamic range of the track in dB.", value=dynamic_range))
            track.metadata.append(AudioMetadataItem(header=Header.Crest_Factor, description="The crest factor of the track.", value=crest_factor))
            track.metadata.append(AudioMetadataItem(header=Header.Bit_Depth, description="The bit depth of the track in bits.", value=file_info.bit_depth))
            track.metadata.append(AudioMetadataItem(header=Header.Max_Data_Per_Second, description="The max data rate per second in bits.", value=max_data_per_second/1000))
            track.metadata.append(AudioMetadataItem(header=Header.Actual_Data_Rate, description="The actual data rate per second in bits.", value=actual_data_rate/1000))
            track.metadata.append(AudioMetadataItem(header=Header.Efficiency, description="The efficiency of data usage.", value=efficiency))
            track.metadata.append(AudioMetadataItem(header=Header.Format, description="The audio format of the track.", value=file_info.format))
            track.metadata.append(AudioMetadataItem(header=Header.Loudness, description="The loudness measurement in LUFS.", value=lufs))
            self._logger.trace(f"Finished adding metadata for track: {track.path}", separator=self._separator)

        self._logger.trace("Finished adding advanced metadata.", separator=self._separator)
        return data
