from pathlib import Path
from typing import List, Tuple

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.data_generation.builders.key_data_frames_builder import \
    KeyDataFramesBuilder
from track_analysis.components.track_analysis.features.data_generation.builders.metadata_df_builder import \
    MetadataDFBuilder
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.feature_to_header_mapping import \
    FEATURE_TO_HEADER_MAPPING
from track_analysis.components.track_analysis.features.data_generation.mappers.results_mapper import ResultsMapper
from track_analysis.components.track_analysis.features.data_generation.processors.key_feature_processor import \
    KeyFeatureProcessor
from track_analysis.components.track_analysis.features.data_generation.processors.main_feature_processor import \
    MainFeatureProcessor
from track_analysis.components.track_analysis.features.data_generation.model.album_cost import AlbumCostModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext


class BatchProcessNewTracks(IPipe):
    """
    Orchestrates the processing of new audio files in batches by delegating
    to specialized processors for each stage of feature extraction.
    """

    def __init__(
            self,
            logger: HoornLogger,
            metadata_builder: MetadataDFBuilder,
            key_data_builder: KeyDataFramesBuilder,
            results_mapper: ResultsMapper,
    ):
        self._logger = logger
        self._separator = "BuildCSV.BatchProcessNewTracks"

        # High-level components this orchestrator manages
        self._metadata_builder = metadata_builder
        self._key_data_builder = key_data_builder
        self._results_mapper = results_mapper

        to_calculate: List[AudioDataFeature] = list(FEATURE_TO_HEADER_MAPPING.keys())
        to_calculate.extend([AudioDataFeature.MFCC_MEANS, AudioDataFeature.MFCC_STDS])

        self._all_features: List[AudioDataFeature] = to_calculate

        self._logger.trace("Initialized batch processor orchestrator.", separator=self._separator)

    def flow(
            self, context: LibraryDataGenerationPipelineContext
    ) -> LibraryDataGenerationPipelineContext:
        """The main execution flow for this pipeline step."""
        paths = context.filtered_audio_file_paths
        batch_size = context.max_new_tracks_per_run

        if not paths:
            self._logger.debug("No new tracks to process.", separator=self._separator)
            self._results_mapper.set_empty_context(context)
            return context

        self._logger.info(
            f"Processing {len(paths)} new tracks in batches of {batch_size}.",
            separator=self._separator,
        )

        # The core processing logic is to run all batches and map the results
        all_results_df, all_key_prog_dfs = self._process_all_batches(context.key_processor, context.main_processor, paths, batch_size, context.album_costs)
        self._results_mapper.map_results_to_context(all_results_df, all_key_prog_dfs, context)

        self._logger.info("Completed batch processing of new tracks.", separator=self._separator)
        return context

    def _process_all_batches(
            self, key_processor: KeyFeatureProcessor, main_processor: MainFeatureProcessor, paths: List[Path], batch_size: int, album_costs: List[AlbumCostModel]
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """Manages the iteration over all batches and concatenates their results."""
        all_batch_results = []
        all_key_progression_dfs = []
        total = len(paths)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_paths = paths[start:end]

            self._logger.info(
                f"Processing batch {start//batch_size + 1}: Tracks {start+1} to {end} of {total}",
                separator=self._separator
            )

            batch_df, key_prog_dfs = self._process_single_batch(key_processor, main_processor, batch_paths, album_costs)
            all_batch_results.append(batch_df)
            all_key_progression_dfs.extend(key_prog_dfs)

        # Concatenate results from all batches into a final DataFrame
        final_df = pd.concat(all_batch_results, ignore_index=True) if all_batch_results else pd.DataFrame()
        return final_df, all_key_progression_dfs

    def _process_single_batch(
            self, key_processor: KeyFeatureProcessor, main_processor: MainFeatureProcessor, batch_paths: List[Path], album_costs: List[AlbumCostModel]
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """
        Handles a single batch by calling a sequence of specialized processors and builders.
        """
        meta_df = self._metadata_builder.build_metadata_df(batch_paths, album_costs)

        raw_key_results = key_processor.extract_raw_keys(meta_df)
        key_df, key_prog_dfs = self._key_data_builder.build(raw_key_results, meta_df)

        meta_with_keys_df = meta_df.merge(key_df, left_index=True, right_on='original_index',
                                          how='left').drop(columns=['original_index'])

        calculated_features_df = main_processor.process_batch(meta_with_keys_df, self._all_features)

        audio_uuid_col = Header.UUID.value
        calculated_features_df_no_dupes = calculated_features_df.drop(columns=[audio_uuid_col])

        final_batch_df = pd.concat(
            [meta_with_keys_df.reset_index(drop=True), calculated_features_df_no_dupes.reset_index(drop=True)],
            axis=1
        )

        return final_batch_df, key_prog_dfs
