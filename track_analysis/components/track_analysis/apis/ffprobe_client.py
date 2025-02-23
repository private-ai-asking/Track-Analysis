import json
import subprocess

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
