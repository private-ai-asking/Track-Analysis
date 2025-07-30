import gc
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
            audio_calculator: AudioCalculator
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

        batch_dfs: List[pd.DataFrame] = []

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_paths = paths[start:end]
            self._process_batch(batch_paths, context, batch_dfs)

        # Concatenate all batch results
        context.generated_audio_info = (
            pd.concat(batch_dfs, ignore_index=True)
            if batch_dfs else pd.DataFrame()
        )

        self._logger.info("Completed batch processing of new tracks.", separator=self._separator)
        return context

    def _process_batch(
            self,
            batch_paths: List[Path],
            context: LibraryDataGenerationPipelineContext,
            batch_dfs: List[pd.DataFrame]
    ) -> None:
        """
        Handle a single batch: extract streams, metadata, and metrics,
        then append the resulting DataFrame to batch_dfs.
        """
        self._logger.info(
            f"Batch {batch_paths[0].name} to {batch_paths[-1].name}: extracting stream info...",
            separator=self._separator
        )

        # 1) Extract stream info (loads samples)
        stream_infos = self._file_handler.get_audio_streams_info_batch(batch_paths)

        # 2) Build basic metadata
        meta_df = self._build_metadata_df(batch_paths, stream_infos, context)

        # 3) Compute advanced audio metrics
        metrics_df = self._build_metrics_df(stream_infos, meta_df)

        # 4) Combine
        result_df = meta_df.join(metrics_df, how="left")
        batch_dfs.append(result_df)

        # 5) Cleanup
        del stream_infos, meta_df, metrics_df, result_df
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
            record = {Header.Audio_Path.value: path}
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

    def _build_metrics_df(
            self,
            infos: List[AudioStreamsInfoModel],
            meta_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Computes and merges sample-based and stream-based audio metrics."""
        samples_list = [info.samples for info in infos]
        sample_rates = [info.sample_rate_Hz for info in infos]

        sample_metrics = self._audio_calculator.calculate_batch_sample_metrics(
            samples_list, sample_rates
        )
        rest_metrics = self._audio_calculator.calculate_batch_rest(
            infos, meta_df[Header.Audio_Path.value].tolist()
        )

        idx = meta_df.index.values
        sample_df = pd.DataFrame({"idx": idx, **sample_metrics}).set_index("idx")
        rest_df = pd.DataFrame({"idx": idx, **rest_metrics}).set_index("idx")

        return sample_df.join(rest_df, how="left")
