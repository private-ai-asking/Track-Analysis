from pathlib import Path
from typing import Union

import mutagen

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.model.audio_info_model import AudioInfoModel


class TagExtractor:
    """
    Utility to extract mp3 tags from an audio file.
    """

    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._module_separator: str = "TagExtractor"
        self._logger.trace("Successfully initialized TagExtractor.", separator=self._module_separator)

    def _load_file(self, file_path: Path) -> Union[mutagen.File, None]:
        self._logger.trace(f"Loading file from: {file_path}.", separator=self._module_separator)

        try:
            file = mutagen.File(str(file_path))
            self._logger.trace(f"Successfully loaded file: {file_path}.", separator=self._module_separator)
            return file
        except mutagen.MutagenError as e:
            self._logger.error(f"Error loading file {file_path}: {e}")
            return None

    def extract(self, audio_file: Path) -> AudioInfoModel:
        """
        Extracts mp3 tags from the given audio file.

        Args:
            audio_file (Path): The path to the audio file.

        Returns:
            AudioInfoModel: An object containing the extracted mp3 tags.
        """
        self._logger.trace(f"Extracting mp3 tags from {audio_file}")

        file: mutagen.File = self._load_file(audio_file)

        if file:
            for key, value in file.items():
                self._logger.debug(f"{key}: {value}", separator=self._module_separator)

        exit()

        self._logger.trace(f"Finished extracting mp3 tags from {audio_file}")
        return AudioInfoModel()
