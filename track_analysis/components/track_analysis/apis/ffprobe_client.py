import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

from track_analysis.components.md_common_python.py_common.logging import HoornLogger


class FFprobeClient:
    """Encapsulates the execution of ffprobe commands."""

    def __init__(self, logger: HoornLogger):
        self._separator = "FFProbeClient"
        self._logger = logger

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
