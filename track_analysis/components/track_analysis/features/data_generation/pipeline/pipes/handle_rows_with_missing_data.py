import gc
from pathlib import Path
from typing import List, Dict, Callable

import numpy as np
import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.audio_calculation.calculators.rms_calculator import \
    compute_short_time_rms_dbfs
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler, AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.core.cacheing.harmonic import HarmonicExtractor
from track_analysis.components.track_analysis.features.core.cacheing.magnitude_spectogram import \
    MagnitudeSpectrogramExtractor
from track_analysis.components.track_analysis.features.core.cacheing.onset_envelope import OnsetStrengthExtractor
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext


class HandleRowsWithMissingData(IPipe):
    def __init__(self, logger: HoornLogger, file_handler: AudioFileHandler):
        self._separator = "BuildCSV.HandleRowsWithMissingData"
        self._file_handler = file_handler

        self._hop_length = 512
        self._onset_extractor = OnsetStrengthExtractor(logger)
        self._harmonic_extractor = HarmonicExtractor(logger)
        self._magnitude_extractor = MagnitudeSpectrogramExtractor(logger)

        self._header_processor_func_mapping: Dict[Header, Callable[[List[str], LibraryDataGenerationPipelineContext], None]] = {
            Header.Bit_Depth: self._handle_missing_bit_depth,
            Header.Start_Key: self._handle_missing_segment_keys,
            Header.End_Key: self._handle_missing_segment_keys,
            Header.Mean_RMS: self._handle_missing_rms,
            Header.Max_RMS: self._handle_missing_rms,
            Header.Percentile_90_RMS: self._handle_missing_rms,
            Header.RMS_IQR: self._handle_missing_rms,
            # Header.Onset_Rate_Notes: self._handle_missing_onset_rate_events  # TODO - Add
        }

        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        for header, uuids in data.missing_headers.items():
            processor = self._header_processor_func_mapping.get(header)
            if processor is None:
                self._logger.warning(
                    f"Header {header} not found, unsupported refill... skipping.",
                    separator=self._separator
                )
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
        audio_infos: List[AudioStreamsInfoModel] = self._file_handler.get_audio_streams_info_batch(paths)
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

    # noinspection t
    def _handle_missing_segment_keys(self, uuids: List[str], data: LibraryDataGenerationPipelineContext) -> None:
        df: pd.DataFrame = data.loaded_audio_info_cache

        # ——— Ensure Start_Key and End_Key exist as object columns ———
        for col in (Header.Start_Key.value, Header.End_Key.value):
            if col not in df.columns:
                df[col] = ""
            else:
                df[col] = df[col].astype(object)

        key_csv_path: Path = Path(data.key_progression_output_file_path)
        if not key_csv_path.exists():
            self._logger.warning(
                f"Key progression CSV not found at {key_csv_path}; cannot fill start/end keys.",
                separator=self._separator
            )
            return

        # Load existing key progression CSV
        segments_df = pd.read_csv(key_csv_path)
        filtered = segments_df.loc[segments_df["Track UUID"].isin(uuids)]

        if filtered.empty:
            self._logger.info(
                "No segments found for given UUIDs; nothing to fill.",
                separator=self._separator
            )
            return

        # Group by UUID to find first and last segments
        grouped = filtered.groupby("Track UUID")

        for uuid, group in grouped:
            # Find row with minimum "Segment Start" for first key
            first_row = group.loc[group["Segment Start"].idxmin()]
            first_key = first_row["Segment Key"]
            # Find row with maximum "Segment End" for last key
            last_row = group.loc[group["Segment End"].idxmax()]
            last_key = last_row["Segment Key"]

            # Find index in main DataFrame for this UUID
            idxs = df.index[df[Header.UUID.value] == uuid].tolist()
            if not idxs:
                continue
            idx = idxs[0]

            # Only fill if missing in df
            if pd.isna(df.loc[idx, Header.Start_Key.value]) or df.loc[idx, Header.Start_Key.value] == "":
                df.loc[idx, Header.Start_Key.value] = first_key
            if pd.isna(df.loc[idx, Header.End_Key.value]) or df.loc[idx, Header.End_Key.value] == "":
                df.loc[idx, Header.End_Key.value] = last_key

        self._logger.info(
            f"Filled start/end keys for {len(grouped)} tracks from key progression CSV.",
            separator=self._separator
        )

    def _handle_missing_rms(self, uuids: List[str], data: LibraryDataGenerationPipelineContext) -> None:
        df: pd.DataFrame = data.loaded_audio_info_cache
        batch_size = data.max_new_tracks_per_run

        # Select rows needing RMS (mean, max or 90th missing)
        mask = df[Header.UUID.value].isin(uuids)
        rows = df.loc[mask]
        total = len(rows)
        self._logger.info(
            f"Found {total} tracks with missing RMS stats.",
            separator=self._separator
        )

        # Process in batches to limit memory usage
        for start in range(0, total, batch_size):
            end = start + batch_size
            rows_chunk = rows.iloc[start:end]
            paths = [Path(p) for p in rows_chunk[Header.Audio_Path.value].tolist()]
            if not paths:
                continue

            # Load audio info and samples for this batch
            stream_infos: List[AudioStreamsInfoModel] = \
                self._file_handler.get_audio_streams_info_batch(paths)

            # Compute RMS stats per track
            stats = [compute_short_time_rms_dbfs(info.samples, info.sample_rate_Hz)
                     for info in stream_infos]
            mean_vals, max_vals, p90_vals, rms_iqr_vals = zip(*stats)

            # Write back into DataFrame for this chunk
            idxs = rows_chunk.index
            df.loc[idxs, Header.Mean_RMS.value] = np.array(mean_vals, dtype=np.float64)
            df.loc[idxs, Header.Max_RMS.value]  = np.array(max_vals,  dtype=np.float64)
            df.loc[idxs, Header.Percentile_90_RMS.value] = np.array(p90_vals, dtype=np.float64)
            df.loc[idxs, Header.RMS_IQR.value] = np.array(rms_iqr_vals, dtype=np.float64)

            self._logger.info(
                f"Filled RMS stats for tracks {start+1} to {min(end, total)}.",
                separator=self._separator
            )

            # Cleanup to free memory
            del stream_infos, stats
            gc.collect()
