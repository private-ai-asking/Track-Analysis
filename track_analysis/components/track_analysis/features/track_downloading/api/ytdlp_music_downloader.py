import os.path
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

import yt_dlp
from yt_dlp.postprocessor import PostProcessor

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
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
    def __init__(self, logger: HoornLogger, output_directory: Path, cookies_file: Path, ffmpeg_path: Path):
        super().__init__(is_child=True)
        self._logger = logger
        self._separator: str = "YTDLPMusicDownloader"

        self._output_dir: Path = output_directory
        self._cookies_file: Path = cookies_file
        self._ffmpeg_path: Path = ffmpeg_path

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

    def _download_urls(self, urls: List[str], max_workers: int = 20) -> Dict[str, Path]:
        """
        Download multiple URLs in parallel, with retries and backoff.

        :param urls: List of video URLs to download
        :param max_workers: Number of tracks to download concurrently
        :return: Mapping from URL to downloaded file Path
        """
        # Base options (will be shallow-copied per-task)
        base_opts = {
            'format': 'ba',
            'postprocessors': [{'key': 'FFmpegExtractAudio'}],
            'noplaylist': True,
            'cookiefile': str(self._cookies_file),
            'ffmpeg_location': str(self._ffmpeg_path),
        }

        def download_one(url: str) -> (str, Path):
            """
            Worker function to download a single URL with retry logic.
            Returns (url, path) or (url, None) on failure.
            """
            retries = 3
            backoff = 2
            for attempt in range(retries):
                try:
                    # 1) Create a fresh YDL for info-extraction
                    ydl_info = yt_dlp.YoutubeDL(base_opts)
                    info = ydl_info.extract_info(url, download=False)
                    raw_title = info.get('title', 'audio')
                    clean_title = self._clean_filename(raw_title)

                    # 2) Build output template for this title
                    outtmpl = os.path.join(self._output_dir, f'{clean_title}.%(ext)s')
                    opts = dict(base_opts, outtmpl=outtmpl)

                    # 3) Create a new YDL for actual download + postprocessing
                    ydl_dl = yt_dlp.YoutubeDL(opts)
                    pp_helper = PostProcessorHelper(self._logger)
                    ydl_dl.add_post_processor(pp_helper)

                    ydl_dl.download([url])
                    downloaded_path = Path(pp_helper.get_last_path())
                    self._logger.info(f'Downloaded track "{clean_title}"', separator=self._separator)
                    time.sleep(0.5)
                    return url, downloaded_path

                except yt_dlp.utils.DownloadError as e:
                    wait = backoff ** attempt
                    self._logger.warning(
                        f"Error downloading '{url}' (attempt {attempt+1}/{retries}): {e}. "
                        f"Retrying in {wait}s...", separator=self._separator
                    )
                    time.sleep(wait)
                except Exception as e:
                    self._logger.error(f"Unexpected error for '{url}': {e}", separator=self._separator)
                    break
            return url, None

        downloaded: Dict[str, Path] = {}
        # Kick off parallel downloads
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(download_one, url): url for url in urls}
            for fut in as_completed(futures):
                url, path = fut.result()
                if path:
                    downloaded[url] = path

        return downloaded

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
