import csv
from pathlib import Path
from typing import List

from track_analysis.components.md_common_python.py_common.cli_framework import CommandLineInterface
from track_analysis.components.md_common_python.py_common.handlers import FileHandler
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.user_input.user_input_helper import UserInputHelper
from track_analysis.components.track_analysis.constants import ROOT_MUSIC_LIBRARY, OUTPUT_DIRECTORY
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor
from track_analysis.components.track_analysis.model.audio_info import AudioInfo


class App:
    def __init__(self, logger: HoornLogger):
        self._user_input_helper: UserInputHelper = UserInputHelper(logger)
        self._tag_extractor: TagExtractor = TagExtractor(logger)
        self._file_handler: FileHandler = FileHandler()
        self._logger = logger

    def run(self):
        cmd: CommandLineInterface = CommandLineInterface(self._logger)
        cmd.add_command(["extract_tags_debug", "etd"], "Debugs the extract tags function.", self._debug_extract_tags)
        cmd.add_command(["make_csv", "mc"], "Makes a CSV file from the extracted metadata.", self._make_csv)
        cmd.start_listen_loop()

    def _debug_extract_tags(self):
        def _always_true_validator(_: str) -> (bool, str):
            return True, ""

        # W:\media\music\[02] organized\[01] hq\Classical\Ludovico Einaudi\Elegy For The Arctic\01 Elegy for the Arctic.flac
        file_to_check: str = self._user_input_helper.get_user_input("Please enter the path to the audio file you want to extract:", str, validator_func=_always_true_validator)

        result: AudioInfo =self._tag_extractor.extract(Path(file_to_check))

        for metadata_item in result.metadata:
            self._logger.info(f"{metadata_item.header} - {metadata_item.description}: {metadata_item.value}")

    def _make_csv(self):
        track_paths = self._file_handler.get_children_paths_fast(ROOT_MUSIC_LIBRARY, ".flac", recursive=True)
        audio_info: List[AudioInfo] = []

        for track_path in track_paths:
            audio_info.append(self._tag_extractor.extract(track_path))

        # 1. Extract all unique headers to determine CSV columns:
        all_headers = set()
        for track_metadata in audio_info[0].metadata:
            all_headers.add(track_metadata.header)

        headers = sorted(list(all_headers))

        with open(OUTPUT_DIRECTORY.joinpath("data.csv"), 'w', newline='', encoding='utf-8') as csvfile:  # Added UTF-8 encoding
            writer = csv.writer(csvfile)

            writer.writerow(headers)

            for track in audio_info:
                track_metadata = track.metadata

                row = []
                # Create a dictionary to quickly lookup value given header
                header_value_map = {item.header: item.value for item in track_metadata}
                for header in headers:
                    row.append(header_value_map.get(header, "")) # Default to empty string if header missing
                writer.writerow(row)

