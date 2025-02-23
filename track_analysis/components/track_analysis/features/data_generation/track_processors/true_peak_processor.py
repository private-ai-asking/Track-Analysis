from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.audio_calculator import AudioCalculator
from track_analysis.components.track_analysis.features.audio_file_handler import AudioStreamsInfoModel, AudioFileHandler
from track_analysis.components.track_analysis.features.data_generation.track_processor_interface import \
    ITrackProcessorStrategy
from track_analysis.components.track_analysis.model.audio_info import AudioInfo
from track_analysis.components.track_analysis.model.audio_metadata_item import AudioMetadataItem
from track_analysis.components.track_analysis.model.header import Header


class TruePeakTrackProcessor(ITrackProcessorStrategy):
    def __init__(self, logger: HoornLogger, audio_file_handler: AudioFileHandler, audio_calculator: AudioCalculator):
        self._audio_file_handler = audio_file_handler
        self._audio_calculator = audio_calculator

        super().__init__(logger, "TruePeak", is_child=True)

    def process_track(self, track: AudioInfo) -> AudioInfo:
        """Processes a single track and returns the updated AudioInfo."""
        self._logger.trace(f"Processing track: {track.get_printed()}")
        file_info: AudioStreamsInfoModel = self._audio_file_handler.get_audio_streams_info(track.path)
        true_peak = self._audio_calculator.calculate_true_peak(file_info.sample_rate_Hz, file_info.samples_librosa)

        updated_metadata = []

        for metadata_item_original in track.metadata:
            updated_metadata.append(metadata_item_original.model_copy(deep=True))

        updated_metadata.append(AudioMetadataItem(header=Header.True_Peak, description="", value=true_peak))
        self._logger.trace(f"Processed track: {track.get_printed()}")
        return AudioInfo(path=track.path, metadata=updated_metadata, timeseries_data=track.timeseries_data)
