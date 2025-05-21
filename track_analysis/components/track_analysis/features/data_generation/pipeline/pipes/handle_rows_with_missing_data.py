from pathlib import Path
from typing import List, Dict, Callable

import numpy as np
import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler, AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext


class HandleRowsWithMissingData(IPipe):
    def __init__(self, logger: HoornLogger, file_handler: AudioFileHandler):
        self._separator = "BuildCSV.HandleRowsWithMissingData"
        self._file_handler = file_handler

        self._header_processor_func_mapping: Dict[Header, Callable[[List[str], LibraryDataGenerationPipelineContext], None]] = {
            Header.Bit_Depth: self._handle_missing_bit_depth
        }

        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        for header, uuids in data.missing_headers.items():
            processor = self._header_processor_func_mapping.get(header, None)
            if processor is None:
                self._logger.warning(f"Header {header} not found, unsupported refill... skipping.", separator=self._separator)
                continue
            processor(uuids, data)

        return data

    def _handle_missing_bit_depth(self, uuids: List[str], data: LibraryDataGenerationPipelineContext) -> None:
        df: pd.DataFrame = data.loaded_audio_info_cache

        # 1) select only rows whose UUID is in uuids **and** whose extension is “.flac”
        rows: pd.DataFrame = df.loc[
            df[Header.UUID.value].isin(uuids) &
            (df[Header.Extension.value].str.lower() == ".flac")
            ]

        self._logger.info(
            f"Found {len(rows)} .flac tracks with missing bit depth.",
            separator=self._separator
        )

        # 2) build Path list (up to your max per run)
        paths: List[Path] = [
                                Path(p) for p in rows[Header.Audio_Path.value].tolist()
                            ][: data.max_new_tracks_per_run]

        # 3) fetch new bit depths
        audio_infos: List[AudioStreamsInfoModel] = (
            self._file_handler.get_audio_streams_info_batch(paths)
        )
        bit_depths: np.ndarray = np.array(
            [i.bit_depth for i in audio_infos],
            dtype=np.float64
        )

        # 4) write back into df at the correct indices
        idxs = rows.index[: len(bit_depths)]
        df.loc[idxs, Header.Bit_Depth.value] = bit_depths

        self._logger.info(
            f"Finished processing {len(paths)} .flac tracks with missing bit depths!",
            separator=self._separator
        )
