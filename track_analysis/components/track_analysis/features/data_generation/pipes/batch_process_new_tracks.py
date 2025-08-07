from pathlib import Path
from typing import List

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.builders.metadata_df_builder import \
    MetadataDFBuilder
from track_analysis.components.track_analysis.features.data_generation.mappers.results_mapper import ResultsMapper
from track_analysis.components.track_analysis.features.data_generation.model.album_cost import AlbumCostModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.data_generation.processors.main_feature_processor import \
    MainFeatureProcessor
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature, MFCC_UNIQUE_FILE_FEATURES, KEY_PROGRESSION_UNIQUE_FILE_FEATURES
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.feature_to_header_mapping import \
    FEATURE_TO_HEADER_MAPPING


class BatchProcessNewTracks(IPipe):
    """
    Orchestrates the processing of new audio files in batches by delegating
    to specialized processors for each stage of feature extraction.
    """

    def __init__(
            self,
            logger: HoornLogger,
            metadata_builder: MetadataDFBuilder,
            results_mapper: ResultsMapper,
    ):
        self._logger = logger
        self._separator = "BuildCSV.BatchProcessNewTracks"

        # High-level components this orchestrator manages
        self._metadata_builder = metadata_builder
        self._results_mapper = results_mapper

        to_retrieve: List[AudioDataFeature] = list(FEATURE_TO_HEADER_MAPPING.keys())
        to_retrieve.extend(MFCC_UNIQUE_FILE_FEATURES)
        to_retrieve.extend(KEY_PROGRESSION_UNIQUE_FILE_FEATURES)

        self._all_features: List[AudioDataFeature] = to_retrieve

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
        all_results_df = self._process_all_batches(context.main_processor, paths, batch_size, context.album_costs)
        self._results_mapper.map_results_to_context(all_results_df, context)

        self._logger.info("Completed batch processing of new tracks.", separator=self._separator)
        return context

    def _process_all_batches(
            self, main_processor: MainFeatureProcessor, paths: List[Path], batch_size: int, album_costs: List[AlbumCostModel]
    ) -> pd.DataFrame:
        """Manages the iteration over all batches and concatenates their results."""
        all_batch_results = []
        total = len(paths)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_paths = paths[start:end]

            self._logger.info(
                f"Processing batch {start//batch_size + 1}: Tracks {start+1} to {end} of {total}",
                separator=self._separator
            )

            batch_df = self._process_single_batch(main_processor, batch_paths, album_costs)
            all_batch_results.append(batch_df)

        # Concatenate results from all batches into a final DataFrame
        final_df = pd.concat(all_batch_results, ignore_index=True) if all_batch_results else pd.DataFrame()
        return final_df

    def _process_single_batch(
            self, main_processor: MainFeatureProcessor, batch_paths: List[Path], album_costs: List[AlbumCostModel]
    ) -> pd.DataFrame:
        # 1. Build the initial metadata DataFrame.
        meta_df = self._metadata_builder.build_metadata_df(batch_paths, album_costs)

        # 2. Retrieve features for the batch. The row order MUST be preserved.
        retrieved_features_df = main_processor.process_batch(meta_df, self._all_features)

        # 3. Identify the UUID column, which likely exists in both DataFrames.
        audio_uuid_col = Header.UUID.value

        # 4. To prevent a duplicate column, drop the UUID from the features DataFrame before concatenating.
        features_only_df = retrieved_features_df.drop(columns=[audio_uuid_col])

        # 5. Concatenate the metadata and features side-by-side (axis=1).
        # .reset_index(drop=True) is a safeguard to ensure the DataFrames align perfectly by row position.
        final_batch_df = pd.concat(
            [meta_df.reset_index(drop=True), features_only_df.reset_index(drop=True)],
            axis=1
        )

        return final_batch_df
