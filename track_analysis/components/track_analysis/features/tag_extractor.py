from pathlib import Path
from pprint import pprint
from typing import Union, List

import librosa
import mutagen
import numpy as np
import pydantic
from mutagen import flac
from mutagen.flac import FLAC

from track_analysis.components.md_common_python.py_common.logging import HoornLogger

from track_analysis.components.track_analysis.model.audio_info import AudioInfo
from track_analysis.components.track_analysis.model.audio_metadata_item import AudioMetadataItem


class StreamInfoModel(pydantic.BaseModel):
    duration: float
    bitrate: float
    sample_rate: float


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
            self._logger.warning(f"Unable to read stream info from audio: {audio_file}", separator=self._module_separator)
            return StreamInfoModel(duration=0, bitrate=0, sample_rate=0)

    def _get_artists(self, file: mutagen.File) -> List[str]:
        artists = file.get('artists', "Unknown")
        if artists == "Unknown":
            artists = file.get('artist', "Unknown")
        if artists == "Unknown":
            artists = file.get('albumartist', ["Unknown"])

        return artists

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

    def _format_list(self, ls: List[str]):
        return ", ".join(ls)

    def _calculate_dynamic_range(self, audio_file: Path) -> float:
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
            return 0

        peak_amplitude = np.max(np.abs(y))
        rms_amplitude = np.sqrt(np.mean(y**2))

        if rms_amplitude == 0:  # Avoid division by zero
            return float('inf') if peak_amplitude > 0 else -float('inf')
        dynamic_range = 20 * np.log10(peak_amplitude / rms_amplitude)

        print(dynamic_range)
        exit()

        return dynamic_range

    def extract(self, audio_file: Path) -> AudioInfo:
        """
        Extracts mp3 tags from the given audio file.

        Args:
            audio_file (Path): The path to the audio file.

        Returns:
            AudioMetadataItem: An object containing the extracted mp3 tags.
        """
        self._logger.trace(f"Extracting mp3 tags from {audio_file}", separator=self._module_separator)

        metadata: List[AudioMetadataItem] = []

        file: mutagen.File = self._load_file(audio_file)
        file_info: StreamInfoModel = self._get_stream_info(audio_file)

        artists = self._get_artists(file)
        artists = self._format_list(artists)

        album_artists = file.get('albumartist', ["Unknown"])
        album_artists = self._format_list(album_artists)

        dynamic_range = self._calculate_dynamic_range(audio_file)

        print(dynamic_range)
        exit()

        # Basic Metadata
        metadata.append(AudioMetadataItem(header="Title", description="The track title.", value=file.get('title', ["Unknown"])[0]))
        metadata.append(AudioMetadataItem(header="Album", description="The album where this track is part of.", value=file.get('album', ["Unknown"])[0]))
        metadata.append(AudioMetadataItem(header="Artist(s)", description="The track artists.", value=artists))
        metadata.append(AudioMetadataItem(header="Album Artist(s)", description="The album artists.", value=album_artists))
        metadata.append(AudioMetadataItem(header="Label", description="The label associated with the track.", value=file.get('label', ["Unknown"])[0]))

        metadata.append(AudioMetadataItem(header="Release Year", description="The original year this track was released.", value=file.get('originalyear', ["Unknown"])[0]))
        metadata.append(AudioMetadataItem(header="Release Date", description="The original date this track was released.", value=file.get('originaldate', ["Unknown"])[0]))

        metadata.append(AudioMetadataItem(header="Genre", description="The genre of the music.", value=file.get('genre', ["Unknown"])[0]))

        # Sonic Metadata
        metadata.append(AudioMetadataItem(header="BPM", description="The tempo of the track.", value=file.get('bpm', ["Unknown"])[0]))
        metadata.append(AudioMetadataItem(header="Energy Level", description="The energy level of the track.", value=file.get('energylevel', ["Unknown"])[0]))
        metadata.append(AudioMetadataItem(header="Key", description="The camelot key of the track.", value=file.get('initialkey', ["Unknown"])[0]))

        # Stream Info
        metadata.append(AudioMetadataItem(header="Duration", description="The duration of the track in seconds.", value=file_info.duration))
        metadata.append(AudioMetadataItem(header="Bitrate", description="The bitrate of the track in kbps.", value=file_info.bitrate))
        metadata.append(AudioMetadataItem(header="Sample Rate", description="The sample rate of the track in Hz.", value=file_info.sample_rate))
        metadata.append(AudioMetadataItem(header="Dynamic Range", description="The peak-to-RMS dynamic range of the track in dB.", value=dynamic_range))

        self._logger.trace(f"Finished extracting mp3 tags from {audio_file}", separator=self._module_separator)
        return AudioInfo(metadata=metadata)
