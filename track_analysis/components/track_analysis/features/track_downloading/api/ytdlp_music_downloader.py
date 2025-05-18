import os.path
import re
import time
from pathlib import Path
from typing import List, Dict

import yt_dlp
from yt_dlp.postprocessor import PostProcessor

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import COOKIES_FILE, FFMPEG_PATH
from track_analysis.components.track_analysis.features.track_downloading.api.music_download_interface import \
    MusicDownloadInterface
from track_analysis.components.track_analysis.features.track_downloading.model.download_model import DownloadModel


class PostProcessorHelper(PostProcessor):
    def __init__(self, logger: HoornLogger):
        super().__init__()
        self._logger = logger
        self._separator: str = "YTDLPMusicDownloader.PostProcessorHelper"

        self._latest_path = ""

    def run(self, information):
        self._latest_path = information["filepath"]
        self._logger.debug(f"Extracted path: '{self._latest_path}'", separator=self._separator)
        return [], information

    def get_last_path(self) -> str:
        self._logger.debug(f"Returning path: '{self._latest_path}'", separator=self._separator)
        return self._latest_path


class YTDLPMusicDownloader(MusicDownloadInterface):
    def __init__(self, logger: HoornLogger, output_directory: Path):
        super().__init__(is_child=True)
        self._logger = logger
        self._separator: str = "YTDLPMusicDownloader"

        self._output_dir: Path = output_directory

        self._logger.debug("YTDLPMusicDownloader initialized", separator=self._separator)

    def download_tracks(self, tracks_to_download: List[DownloadModel]) -> List[DownloadModel]:
        return self._download_tracks(tracks_to_download)

    def _download_tracks(self, tracks_to_download: List[DownloadModel]) -> List[DownloadModel]:
        track_urls: List[str] = [track.url for track in tracks_to_download]
        paths: Dict[str, Path] = self._download_urls(track_urls)
        download_models: List[DownloadModel] = [track.model_copy(deep=True) for track in tracks_to_download]

        url_data: Dict[str, DownloadModel] = {track.url: track for track in tracks_to_download}

        for url, path in paths.items():
            self._add_path_to_track_model(url, path, download_models)

        missed_urls = [url for url in track_urls if url not in paths.keys()]
        for url in missed_urls:
            self._logger.warning(f"The following url could not be downloaded: {url}, id: {url_data[url].recording_id}", separator=self._separator)

        return download_models

    def _add_path_to_track_model(self, url: str, path: Path, tracks: List[DownloadModel]) -> None:
        for track in tracks:
            if track.url == url:
                track.path = path
                return

        self._logger.warning(f"No path could be found for url: \"{url}\"", separator=self._separator)

    def _download_urls(self, urls: List[str]) -> Dict[str, Path]:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio'
            }],
            'outtmpl': os.path.join(self._output_dir, '%(title)s.%(ext)s'),
            'noplaylist': True,
            "cookiefile": str(COOKIES_FILE),
            "ffmpeg_location": str(FFMPEG_PATH)
        }

        downloaded_files: Dict[str, Path] = {}

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            post_processor_helper = PostProcessorHelper(self._logger)
            ydl.add_post_processor(post_processor_helper)

            for url in urls:
                retries = 3
                backoff_factor = 2
                for i in range(retries):
                    try:
                        # Step 1: Pre-fetch to clean the title
                        info_dict = ydl.extract_info(url, download=False)
                        title = info_dict.get('title', 'audio')
                        title = self._clean_filename(title)

                        # Step 2: Set custom output filename template
                        ydl_opts['outtmpl']['default'] = os.path.join(self._output_dir, f'{title}.%(ext)s')

                        ydl.download([url])
                        downloaded_files[url] = Path(post_processor_helper.get_last_path())

                        self._logger.info(f"Downloaded track \"{title}\"", separator=self._separator)

                        time.sleep(0.5)
                        break  # Comment this line if you want to torment your soul for all eternity.
                    except yt_dlp.utils.DownloadError as e:
                        wait_time = backoff_factor ** i
                        self._logger.warning(f"Error downloading '{url}': {e}, retrying in {wait_time} seconds...", separator=self._separator)
                        time.sleep(wait_time)
                    except Exception as e:
                        self._logger.error(f"An error occurred while downloading '{url}': {e}", separator=self._separator)
                        break

        return downloaded_files

    def _clean_filename(self, filename: str, replacement_char: str = '_') -> str:
        """
        Cleans a filename by:
        - Replacing unsupported or non-UTF-8-safe characters
        - Replacing invalid filename characters
        - Removing leading/trailing spaces or dots

        Args:
            filename: The filename to clean.
            replacement_char: The character to replace unsupported characters with.

        Returns:
            The cleaned filename.
        """
        # Replace characters not encodable in UTF-8
        safe_chars = []
        for char in filename:
            try:
                char.encode('utf-8')
                safe_chars.append(char)
            except UnicodeEncodeError:
                safe_chars.append(replacement_char)
        filename = ''.join(safe_chars)

        # Define a regular expression for invalid filename characters
        invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'

        # Replace them with the replacement character
        filename = re.sub(invalid_chars, replacement_char, filename)

        # Replace non-printable characters
        filename = ''.join(
            c if c.isprintable() else replacement_char
            for c in filename
        )

        # Remove leading and trailing dots and spaces
        filename = filename.strip(' .')

        # Ensure it's not empty
        if not filename:
            filename = replacement_char

        return filename
