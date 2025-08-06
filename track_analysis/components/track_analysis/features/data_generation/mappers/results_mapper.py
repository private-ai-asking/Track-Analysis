from typing import Dict, List

import pandas as pd

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.data_generation.builders.key_progression_df_builder import \
    KeyProgressionDFBuilder
from track_analysis.components.track_analysis.features.data_generation.builders.main_audio_info_df_builder import \
    MainAudioInfoDFBuilder
from track_analysis.components.track_analysis.features.data_generation.builders.mfcc_df_builder import MfccDFBuilder
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext


class ResultsMapper:
    def __init__(self, feature_to_header_map: Dict[AudioDataFeature, Header]):
        self._main_info_builder = MainAudioInfoDFBuilder(feature_to_header_map)
        self._mfcc_builder = MfccDFBuilder()
        self._key_progression_builder = KeyProgressionDFBuilder()

    def map_results_to_context(
            self, full_df: pd.DataFrame, key_prog_dfs: List[pd.DataFrame], context: LibraryDataGenerationPipelineContext
    ) -> None:
        """Orchestrates building final DataFrames and assigning them to the context."""
        if full_df.empty:
            self.set_empty_context(context)
            return

        context.generated_audio_info = self._main_info_builder.build(full_df)
        context.generated_mfcc_audio_info = self._mfcc_builder.build(full_df)
        context.generated_key_progression_audio_info = self._key_progression_builder.build(key_prog_dfs)

    @staticmethod
    def set_empty_context(context: LibraryDataGenerationPipelineContext):
        """Sets empty DataFrames on the context for a clean state."""
        context.generated_audio_info = pd.DataFrame()
        context.generated_mfcc_audio_info = pd.DataFrame()
        context.generated_key_progression_audio_info = pd.DataFrame()
