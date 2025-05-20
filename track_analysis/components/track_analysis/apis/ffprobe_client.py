import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.exceptions.ffprobe_error import FFprobeError


class FFprobeClient:
    """Encapsulates the execution of ffprobe commands."""

    def __init__(self, logger: HoornLogger):
        self._separator = "FFProbeClient"
        self._logger = logger

    def run_ffprobe(self, audio_file_path: str) -> dict:
        """
        Runs ffprobe and returns the output as a dictionary.

        Args:
            audio_file_path: The path to the audio file.

        Returns:
            A dictionary containing the ffprobe output.

        Raises:
            FileNotFoundError: If ffprobe is not found.
            FFprobeError: If ffprobe returns a non-zero exit code or the output is invalid.
        """
        cmd = ["ffprobe", '-v', 'error', '-show_format', '-show_streams', '-of', 'json', audio_file_path]
        self._logger.debug(f"Running command: {' '.join(cmd)}", separator=self._separator)
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", check=True)  # Raises CalledProcessError on non-zero exit code
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError as e:
                self._logger.warning(f"Error decoding JSON from ffprobe output: {e}", separator=self._separator)
                self._logger.warning(f"ffprobe output: {result.stdout}", separator=self._separator)
                raise FFprobeError("Invalid JSON output from ffprobe") from e
        except FileNotFoundError as e:
            self._logger.error("ffprobe not found. Please install FFmpeg.", separator=self._separator)
            raise e
        except subprocess.CalledProcessError as e:
            self._logger.warning(f"ffprobe error: {e.stderr}", separator=self._separator)
            raise FFprobeError(f"ffprobe command failed with return code {e.returncode}") from e

    def run_ffprobe_batch(self, audio_file_paths: List[str]) -> Dict[str, dict]:
        """
        Runs ffprobe separately on each audio file—concurrently if requested—
        and returns a mapping from file path to its ffprobe output dict.

        Args:
            audio_file_paths: A list of paths to audio files.

        Returns:
            A dict mapping each audio_file_path to its ffprobe JSON output.

        Raises:
            Any exception raised by run_ffprobe for a particular file will
            propagate, after being logged.
        """
        outputs: Dict[str, dict] = {}

        def _probe(path: str):
            try:
                info = self.run_ffprobe(path)
                return path, info
            except Exception as e:
                self._logger.warning(
                    f"Error probing '{path}': {e}", separator=self._separator
                )
                raise

        if audio_file_paths:
            # Use threads to parallelize I/O-bound subprocess calls
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(_probe, p): p for p in audio_file_paths}
                total = len(futures)
                for i, future in enumerate(as_completed(futures), start=1):
                    path, info = future.result()
                    outputs[path] = info
                    self._logger.info(
                        f"Probed {i}/{total}: {path}", separator=self._separator
                    )

        return outputs
