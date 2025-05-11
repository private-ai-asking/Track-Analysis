from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import PipelineContextModel


class PreprocessData(IPipe):
    def __init__(self, logger: HoornLogger):
        self._separator = "PreprocessDataPipe"

        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def _standardize_artists(self, ls: List[str]) -> List[str]:
        to_standardize = [
            {
                "To": "Damian \"Jr. Gong\" Marley",
                "From": [
                    "Damian \"Jr. Gong\" Marley",
                    "Damian Jr. Gong Marley",
                    "Damian “Jr. Gong” Marley"
                ]
            }
        ]
        standardized: List[str] = []

        for item in ls:
            for standardize in to_standardize:
                if item in standardize["From"]:
                    item = standardize["To"]
                    break
            standardized.append(item)

        return standardized

    def _format_list(self, ls: List[str]) -> str:
        return ", ".join(ls)

    def flow(self, data: PipelineContextModel) -> PipelineContextModel:
        self._logger.trace("Preprocessing data...", separator=self._separator)

        for track in data.generated_audio_info:
            for metadata_item in track.metadata:
                if metadata_item.header == Header.Artists:
                    self._logger.debug(f"Preprocessing old value: {metadata_item.value}", separator=self._separator)
                    metadata_item.value = self._format_list(self._standardize_artists(metadata_item.value))
                    self._logger.debug(f"Preprocessing new value: {metadata_item.value}", separator=self._separator)
                if metadata_item.header == Header.Album_Artists:
                    self._logger.debug(f"Preprocessing old value: {metadata_item.value}", separator=self._separator)
                    metadata_item.value = self._format_list(metadata_item.value)
                    self._logger.debug(f"Preprocessing new value: {metadata_item.value}", separator=self._separator)

        self._logger.trace("Successfully preprocessed data.", separator=self._separator)
        return data
