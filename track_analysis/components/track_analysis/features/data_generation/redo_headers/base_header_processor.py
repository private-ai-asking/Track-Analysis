from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Callable, Any, Tuple

import mutagen
import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler
from track_analysis.components.track_analysis.features.core.log_utils import log_and_handle_exception
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext


class BaseHeaderProcessor(ABC):
    """
    Abstract base class for all header processors.
    Encapsulates common logic like logging, data collection, and DataFrame updates.
    """
    _SEPARATOR = "BuildCSV.RedoHeaders"

    def __init__(self, logger: HoornLogger, file_handler: AudioFileHandler):
        self._logger = logger
        self._file_handler = file_handler

    @abstractmethod
    def process(self, uuids: List[str], data: LibraryDataGenerationPipelineContext) -> None:
        """
        The main method to be implemented by each concrete processor.
        """
        pass

    def _log_and_handle_exception(self, message: str, action: Callable, *args: Any, **kwargs: Any) -> Any:
        """Helper to wrap an action in a try-except block for consistent error logging."""
        return log_and_handle_exception(self._logger, message, action, *args, **kwargs)

    def _collect_index_path_pairs(self, df: pd.DataFrame, uuids: List[str]) -> List[Tuple[int, Path]]:
        """Collects DataFrame index and audio file path for given UUIDs."""
        rows = df.loc[df[Header.UUID.value].isin(uuids)]
        return [
            (idx, Path(p))
            for idx, p in zip(rows.index, rows[Header.Audio_Path.value].tolist())
        ]

    def _update_dataframe_column(self, df: pd.DataFrame, indices: List[int], column: Header, values: List[Any]):
        """Updates a DataFrame column for given indices and values."""
        self._log_and_handle_exception(
            f"Error updating DataFrame for column '{column.value}'",
            lambda: df.loc[indices, column.value].update(pd.Series(values, index=indices))
        )

    def _write_single_tag(self, path: Path, tag_key: str, tag_value: str):
        """Helper to write a single tag to an audio file."""
        try:
            tag_file = mutagen.File(str(path), easy=True)
            if tag_file:
                tag_file[tag_key] = tag_value
                tag_file.save()
        except Exception:
            raise  # Reraise to be caught by the _log_and_handle_exception wrapper

    def _update_audio_tags(self, paths: List[Path], tag_key: str, tag_values: List[str]):
        """Writes a given tag to audio files."""
        for path, value in zip(paths, tag_values):
            if not isinstance(path, Path) or not path.is_file():
                self._logger.warning(f"Invalid path for tag update: {path}", separator=self._SEPARATOR)
                continue
            self._log_and_handle_exception(
                f"Failed to write tag '{tag_key}' for file '{path}'",
                self._write_single_tag, path, tag_key, value
            )
