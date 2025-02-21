import csv

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.pipeline.pipeline_context import PipelineContextModel


class MakeCSV(IPipe):
    def __init__(self, logger: HoornLogger):
        self._separator = "MakeCSVPipe"

        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: PipelineContextModel) -> PipelineContextModel:
        # 1. Extract all unique headers to determine CSV columns:
        all_headers = set()

        self._logger.trace("Extracting all unique headers...", separator=self._separator)
        for track_metadata in data.audio_info[0].metadata:
            all_headers.add(track_metadata.header)

        headers = sorted(list(all_headers))
        self._logger.trace("Successfully extracted all unique headers.", separator=self._separator)

        self._logger.trace("Writing data...", separator=self._separator)
        with open(data.output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(headers)

            for track in data.audio_info:
                track_metadata = track.metadata

                row = []
                header_value_map = {item.header: item.value for item in track_metadata}
                for header in headers:
                    row.append(header_value_map.get(header, ""))

                self._logger.debug(f"Writing row: {row}", separator=self._separator)
                writer.writerow(row)

        self._logger.trace("Successfully written all data.", separator=self._separator)

        return data
