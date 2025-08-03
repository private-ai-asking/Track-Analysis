import gc
import uuid
from pathlib import Path
from typing import List

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.audio_calculator import AudioCalculator
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler, AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor


class BatchProcessNewTracks(IPipe):
    """
    Processes new audio files in memory-safe batches: extracts stream info,
    tags metadata, computes audio metrics, and concatenates the results.
    """

    def __init__(
            self,
            logger: HoornLogger,
            file_handler: AudioFileHandler,
            tag_extractor: TagExtractor,
            audio_calculator: AudioCalculator,
    ):
        self._logger = logger
        self._file_handler = file_handler
        self._tag_extractor = tag_extractor
        self._audio_calculator = audio_calculator
        self._separator = "BuildCSV.BatchProcessNewTracks"
        self._logger.trace("Initialized batch processor.", separator=self._separator)

    def flow(self, context: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        paths = context.filtered_audio_file_paths
        batch_size = context.max_new_tracks_per_run
        total = len(paths)

        if not paths:
            self._logger.debug("No new tracks to process.", separator=self._separator)
            context.generated_audio_info = pd.DataFrame()
            return context

        self._logger.info(
            f"Processing {total} new tracks in batches of {batch_size}.",
            separator=self._separator
        )

        batch_main_dfs: List[pd.DataFrame] = []
        batch_mfcc_dfs: List[pd.DataFrame] = []
        batch_key_progression_dfs: List[pd.DataFrame] = []

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_paths = paths[start:end]
            self._process_batch(batch_paths, context, batch_main_dfs, batch_mfcc_dfs, batch_key_progression_dfs)

        # Concatenate all batch results
        context.generated_audio_info = (
            pd.concat(batch_main_dfs, ignore_index=True)
            if batch_main_dfs else pd.DataFrame()
        )
        context.generated_mfcc_audio_info = (
            pd.concat(batch_mfcc_dfs, ignore_index=True)
            if batch_mfcc_dfs else pd.DataFrame()
        )
        context.generated_key_progression_audio_info = (
            pd.concat(batch_key_progression_dfs, ignore_index=True)
            if batch_key_progression_dfs else pd.DataFrame()
        )

        context.generated_audio_info = context.energy_calculator.calculate_ratings_for_df(context.generated_audio_info, Header.Energy_Level)

        self._logger.info("Completed batch processing of new tracks.", separator=self._separator)
        return context

    def _process_batch(
            self,
            batch_paths: List[Path],
            context: LibraryDataGenerationPipelineContext,
            batch_main_dfs: List[pd.DataFrame],
            batch_mfcc_dfs: List[pd.DataFrame],
            batch_key_progression_dfs: List[pd.DataFrame],
    ) -> None:
        """
        Handle a single batch: extract streams, metadata, and both main and MFCC metrics,
        then append the results to their respective lists.
        """
        self._logger.info(
            f"Batch {batch_paths[0].name} to {batch_paths[-1].name}: processing...",
            separator=self._separator
        )

        # 1) Get stream info (which includes samples, sr, tempo, etc.)
        stream_infos = self._file_handler.get_audio_streams_info_batch(batch_paths)

        # 2) Build basic metadata DataFrame (Artist, Title, UUID, etc.)
        meta_df = self._build_metadata_df(batch_paths, stream_infos, context)
        uuids: List[str] = meta_df[Header.UUID.value].tolist()

        # 3) Compute audio metrics directly to get both DataFrames
        infos, paths, samples, rates, tempos = zip(*((i, i.path, i.samples, i.sample_rate_Hz, i.tempo) for i in stream_infos))
        processing_result = self._audio_calculator.process(infos, paths, uuids, samples, rates, tempos)

        # 4) Combine metadata with the main metrics using a robust merge on UUID
        # This ensures correct alignment even if the order changes.
        uuid_col = Header.UUID.value
        merged_main_df = pd.merge(meta_df, processing_result.main_df, on=uuid_col, how="left")

        # 5) Append results to their respective lists
        batch_main_dfs.append(merged_main_df)
        batch_mfcc_dfs.append(processing_result.mfcc_df)
        batch_key_progression_dfs.append(processing_result.key_progression_df)

        # 6) Cleanup
        del stream_infos, meta_df, processing_result, merged_main_df
        gc.collect()

    def _build_metadata_df(
            self,
            paths: List[Path],
            infos: List[AudioStreamsInfoModel],
            context: LibraryDataGenerationPipelineContext
    ) -> pd.DataFrame:
        """Creates a DataFrame with file paths and extracted tags."""
        records = []
        for path, info in zip(paths, infos):
            uid = uuid.uuid4()

            record = {
                Header.Audio_Path.value: path,
                Header.UUID.value: str(uid),
            }
            self._tag_extractor.add_extracted_metadata_to_track(record, info)
            records.append(record)

        df = pd.DataFrame(records)

        # Add album costs if available
        if getattr(context, 'album_costs', None):
            cost_map = {c.Album_Title: c.Album_Cost for c in context.album_costs}
            df[Header.Album_Cost.value] = (
                df[Header.Album.value].map(cost_map).fillna(0)
            )

        return df

