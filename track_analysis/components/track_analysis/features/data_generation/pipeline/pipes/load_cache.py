import csv
from pathlib import Path
from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.constants import DEBUG
from track_analysis.components.track_analysis.features.data_generation.model.audio_info import AudioInfo
from track_analysis.components.track_analysis.features.data_generation.model.audio_metadata_item import AudioMetadataItem
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import PipelineContextModel


class LoadCache(IPipe):
    def __init__(self, logger: HoornLogger):
        self._separator = "LoadCachePipe"

        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def _count_csv_rows(self, file_path: Path):
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)
            row_count = sum(1 for _ in reader)
        return row_count

    def flow(self, data: PipelineContextModel) -> PipelineContextModel:
        self._logger.trace("Loading cache if existing...", separator=self._separator)

        if not data.main_data_output_file_path.exists() or not data.main_data_output_file_path.is_file():
            return data

        self._logger.trace("Cache exists, proceeding to load...", separator=self._separator)

        loaded_tracks: List[AudioInfo] = []
        num_lines = self._count_csv_rows(data.main_data_output_file_path)

        with open(data.main_data_output_file_path, "r", newline='', encoding="utf8") as csvfile:
            reader = csv.reader(csvfile)

            header = next(reader)
            header_items: List[Header] = []

            for header_item in header:
                try:
                    header_items.append(Header(header_item))
                    self._logger.trace(f"Correctly parsed header item: {header_item}", separator=self._separator)
                except Exception as e:
                    self._logger.error(f"Error parsing header item: {header_item}. Error: {str(e)}", separator=self._separator)
                    continue

            processed: int = 0

            for row in reader:
                # Map values to headers in a dictionary
                try:
                    track_dict = dict(zip(header_items, row))
                    self._logger.trace(f"Correctly read row: {row}")
                except Exception as e:
                    self._logger.error(f"Error mapping row to dictionary. Error: {str(e)}", separator=self._separator)
                    continue

                try:
                    metadata_items: List[AudioMetadataItem] = []

                    audio_path = None

                    for header, value in track_dict.items():
                        metadata_items.append(AudioMetadataItem(header=header, value=value, description="Irrelevant... (loaded from cache)"))

                        if header == Header.Audio_Path:
                            audio_path = Path(value)

                    if audio_path is None:
                        self._logger.error(f"No audio path found in row: {row}", separator=self._separator)
                        continue

                    audio_info = AudioInfo(metadata=metadata_items, path=audio_path)
                except Exception as e:
                    self._logger.error(f"Error mapping audio info to dictionary. Error: {str(e)}", separator=self._separator)
                    continue

                loaded_tracks.append(audio_info)
                processed += 1
                self._logger.info(f"Successfully loaded track: {processed}/{num_lines} ({round(processed/num_lines * 100, 4)}%)", separator=self._separator)

                if DEBUG and processed >= 10:
                    break

        data.loaded_audio_info_cache = loaded_tracks

        return data
