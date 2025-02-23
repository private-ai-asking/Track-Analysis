import csv
from pathlib import Path
from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.model.audio_info import AudioInfo
from track_analysis.components.track_analysis.model.header import Header
from track_analysis.components.track_analysis.pipeline.pipeline_context import PipelineContextModel


class MakeCSV(IPipe):
    def __init__(self, logger: HoornLogger):
        self._separator = "MakeCSVPipe"

        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def _write_row(self, track: AudioInfo, headers: List[Header], writer: csv.writer):
        track_metadata = track.metadata

        row = []
        header_value_map = {item.header.value: item.value for item in track_metadata}
        for header in headers:
            row.append(header_value_map.get(header, ""))

        self._logger.debug(f"Writing row: {row}", separator=self._separator)
        writer.writerow(row)

    def _extract_headers(self, data: List[AudioInfo], identifier: str) -> List[Header]:
        all_headers = set()

        self._logger.trace("Extracting all unique headers...", separator=self._separator)

        if len(data) <= 0:
            self._logger.warning(f"No audio data found for {identifier}.", separator=self._separator)
            return []

        for track_metadata in data[0].metadata:
            all_headers.add(track_metadata.header.value)

        headers = sorted(list(all_headers))
        self._logger.trace("Successfully extracted all unique headers.", separator=self._separator)
        return headers

    def _write_data(self, writer: csv.writer, data: List[AudioInfo], exclude: List[Path], headers: List[Header]):
        for track in data:
            if track.path in exclude:
                self._logger.debug(f"Skipping invalid cached path: {track.path}", separator=self._separator)
                continue

            self._write_row(track, headers, writer)

    def _write_main_data(self, data: PipelineContextModel):
        with open(data.main_data_output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            headers = self._extract_headers(data.generated_audio_info, "generated audio metadata")

            if len(headers) == 0:
                headers = self._extract_headers(data.loaded_audio_info_cache, "cached audio metadata")

            if len(headers) == 0:
                self._logger.warning("No headers / data found.", separator=self._separator)
                return data

            writer = csv.writer(csvfile)

            writer.writerow(headers)
            self._write_data(writer, data.loaded_audio_info_cache, data.invalid_cached_paths, headers)
            self._write_data(writer, data.generated_audio_info, [], headers)

    def _write_time_series_data(self, context: PipelineContextModel, data: List[AudioInfo]):
        with open(context.main_data_output_file_path.parent.joinpath("timeseries-data.csv"), "w", encoding="utf-8", newline='') as csvfile:
            headers = self._extract_headers(data, "timeseries generated metadata")

            if len(headers) == 0:
                self._logger.warning("No headers / data found.", separator=self._separator)
                return context

            writer = csv.writer(csvfile)

            writer.writerow(headers)
            self._write_data(writer, data, [], headers)

    def flow(self, context: PipelineContextModel) -> PipelineContextModel:
        self._logger.trace("Writing data...", separator=self._separator)

        self._write_main_data(context)

        timeseries_data: List[AudioInfo] = []

        for track in context.generated_audio_info:
            timeseries_data.extend(track.timeseries_data)

        self._write_time_series_data(context, timeseries_data)

        self._logger.trace("Successfully written all data.", separator=self._separator)

        return context
