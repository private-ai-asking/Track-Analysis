from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.md_common_python.py_common.utils import StringUtils
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import (
    LibraryDataGenerationPipelineContext,
)


class PreprocessData(IPipe):
    def __init__(self, logger: HoornLogger, string_utils: StringUtils):
        self._separator = "BuildCSV.PreprocessDataPipe"
        self._logger = logger
        self._string_utils = string_utils
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    @staticmethod
    def _standardize_artists(ls: List[str]) -> List[str]:
        to_standardize = [
            {
                "To": 'Damian "Jr. Gong" Marley',
                "From": [
                    'Damian "Jr. Gong" Marley',
                    "Damian Jr. Gong Marley",
                    'Damian “Jr. Gong” Marley',
                ],
            }
        ]
        standardized: List[str] = []

        for item in ls:
            for rule in to_standardize:
                if item in rule["From"]:
                    item = rule["To"]
                    break
            standardized.append(item)

        return standardized

    @staticmethod
    def _format_list(ls: List[str]) -> str:
        return ", ".join(ls)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        df = data.generated_audio_info
        self._logger.trace("Preprocessing data...", separator=self._separator)

        if len(data.generated_audio_info) <= 0:
            self._logger.debug("Empty generated audio info. Skipping.", separator=self._separator)
            return data

        for idx in df.index:
            # Artists column
            old_artists = df.at[idx, Header.Artists.value]
            self._logger.trace(f"Preprocessing old value: {old_artists}", separator=self._separator)
            new_artists = self._format_list(self._standardize_artists(old_artists))
            df.at[idx, Header.Artists.value] = new_artists
            self._logger.trace(f"Preprocessing new value: {new_artists}", separator=self._separator)

            # Album Artists column
            old_album_artists = df.at[idx, Header.Album_Artists.value]
            self._logger.trace(f"Preprocessing old value: {old_album_artists}", separator=self._separator)
            new_album_artists = self._format_list(self._standardize_artists(old_album_artists))
            df.at[idx, Header.Album_Artists.value] = new_album_artists
            self._logger.trace(f"Preprocessing new value: {new_album_artists}", separator=self._separator)

        df[Header.Primary_Artist.value] = (
            df[Header.Artists.value]
            .apply(self._string_utils.extract_primary_from_sequence)
        )

        self._logger.info("Successfully preprocessed data.", separator=self._separator)
        data.generated_audio_info = df
        return data
