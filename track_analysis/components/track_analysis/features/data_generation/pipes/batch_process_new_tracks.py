from typing import List

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.builders.metadata_df_builder import \
    MetadataDFBuilder
from track_analysis.components.track_analysis.features.data_generation.mappers.results_mapper import ResultsMapper
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext
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

        self._audio_uuid_col = Header.UUID.value

        self._all_features: List[AudioDataFeature] = to_retrieve

        self._logger.trace("Initialized batch processor orchestrator.", separator=self._separator)

    def flow(
            self, context: LibraryDataGenerationPipelineContext
    ) -> LibraryDataGenerationPipelineContext:
        """The main execution flow for this pipeline step."""
        paths = context.filtered_audio_file_paths

        if not paths:
            self._logger.debug("No new tracks to process.", separator=self._separator)
            self._results_mapper.set_empty_context(context)
            return context

        self._logger.info(f"Processing {len(paths)} new tracks.", separator=self._separator)
        meta_df = self._metadata_builder.build_metadata_df(paths, context.album_costs)
        retrieved_features_df = context.main_processor.process_batch(meta_df, self._all_features)
        retrieved_features_df = retrieved_features_df.drop(columns=[self._audio_uuid_col])

        all_results_df = pd.concat(
            [meta_df.reset_index(drop=True), retrieved_features_df.reset_index(drop=True)],
            axis=1
        )

        self._results_mapper.map_results_to_context(all_results_df, context)

        self._logger.info(f"Completed processing of {len(paths)} new tracks.", separator=self._separator)
        return context
