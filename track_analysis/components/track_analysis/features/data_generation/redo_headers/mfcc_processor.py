import time
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.time_handling import TimeUtils
from track_analysis.components.track_analysis.features.audio_calculation.batch_sample_metrics_service import \
    BatchSampleMetricsService
from track_analysis.components.track_analysis.features.audio_calculation.providers.helpers.mfcc_helper import \
    MFCCHelper
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler, AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.core.cacheing.mfcc import MfccExtractor
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.data_generation.redo_headers.base_header_processor import \
    BaseHeaderProcessor


class MFCCProcessor(BaseHeaderProcessor):
    def __init__(self, logger: HoornLogger, file_handler: AudioFileHandler, mfcc_extractor: MfccExtractor):
        super().__init__(logger, file_handler)
        self._time_utils = TimeUtils()
        self._mfcc_helper: MFCCHelper = MFCCHelper(mfcc_extractor)

    @staticmethod
    def _get_existing_tempos(
            chunk: pd.DataFrame
    ) -> Dict[Path, float]:
        """Extracts BPM values for each audio path in this chunk."""
        return {
            Path(p): bpm
            for p, bpm in zip(
                chunk[Header.Audio_Path.value],
                chunk[Header.BPM.value]
            )
        }

    def process(self, uuids: List[str], data: LibraryDataGenerationPipelineContext) -> None:
        """
        Main method to process and update MFCC data for a list of tracks.
        """
        main_df = data.loaded_audio_info_cache
        mfcc_df = data.loaded_mfcc_info_cache
        batch_size = data.max_new_tracks_per_run

        rows_to_process = self._get_rows_to_process(uuids, main_df)
        if rows_to_process.empty:
            self._logger.info("No new tracks to process.", separator=self._SEPARATOR)
            return

        total_rows = len(rows_to_process)

        for start in range(0, total_rows, batch_size):
            end = start + batch_size
            chunk = rows_to_process.iloc[start:end]

            # Process the batch and get the new data and elapsed time
            mfcc_df_chunk, elapsed = self._process_batch(chunk)

            # Update the main MFCC cache with the new batch data
            mfcc_df = self._update_mfcc_cache(mfcc_df, mfcc_df_chunk)
            data.loaded_mfcc_info_cache = mfcc_df

            # Log the progress
            elapsed_formatted = self._time_utils.format_time(elapsed, round_digits=4)
            self._logger.info(
                f"Filled MFCC stats for tracks {start + 1} to {min(end, total_rows)} [duration: {elapsed_formatted}].",
                separator=self._SEPARATOR
            )

    def _get_rows_to_process(self, uuids: List[str], df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the main DataFrame to get rows for the specified UUIDs.
        """
        return df[df[Header.UUID.value].isin(uuids)]

    def _process_batch(self, chunk: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Processes a single chunk of audio tracks to compute their MFCC features.
        """
        start_time = time.time()

        paths = [Path(p) for p in chunk[Header.Audio_Path.value].tolist()]
        if not paths:
            return pd.DataFrame(), 0.0

        existing_tempos = self._get_existing_tempos(chunk)
        stream_infos = self._file_handler.get_audio_streams_info_batch(
            paths,
            existing_tempos=existing_tempos
        )

        mfcc_stats = self._get_mfcc_stats(stream_infos)

        chunk_uuids = chunk[Header.UUID.value].tolist()
        mfcc_means = [s["mfcc_means"] for s in mfcc_stats]
        mfcc_stds = [s["mfcc_stds"] for s in mfcc_stats]

        mfcc_metrics_chunk = {
            Header.UUID.value: chunk_uuids,
            "mfcc_means": mfcc_means,
            "mfcc_stds": mfcc_stds
        }

        mfcc_df_chunk = BatchSampleMetricsService.build_mfcc_dataframe(mfcc_metrics_chunk)
        elapsed = time.time() - start_time

        return mfcc_df_chunk, elapsed

    def _get_mfcc_stats(self, stream_infos: List[AudioStreamsInfoModel]) -> List[dict]:
        """
        Extracts MFCC features from a list of audio stream info models.
        """
        mfcc_stats = []
        for info in stream_infos:
            mfcc_means, mfcc_stds = self._mfcc_helper.get_mffcs(
                audio_path=info.path,
                audio=info.samples,
                sample_rate=info.sample_rate_Hz
            )
            mfcc_stats.append({
                "mfcc_means": mfcc_means,
                "mfcc_stds": mfcc_stds
            })
        return mfcc_stats

    @staticmethod
    def _update_mfcc_cache(current_cache: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merges new MFCC data into the existing MFCC cache, replacing old values.
        """
        if new_data.empty:
            return current_cache

        if current_cache.empty:
            return new_data

        merged_df = pd.merge(
            current_cache,
            new_data,
            on=Header.UUID.value,
            how='outer',
            suffixes=('_old', '_new'),
            indicator=True
        )

        for col in new_data.columns:
            if col != Header.UUID.value:
                merged_df[col] = merged_df[f"{col}_new"].combine_first(merged_df[f"{col}_old"])

        columns_to_drop = [c for c in merged_df.columns if c.endswith(('_old', '_new')) or c == '_merge']
        return merged_df.drop(columns=columns_to_drop)
