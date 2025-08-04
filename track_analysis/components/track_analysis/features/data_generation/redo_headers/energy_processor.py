from typing import List

from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.data_generation.redo_headers.base_header_processor import \
    BaseHeaderProcessor


class EnergyProcessor(BaseHeaderProcessor):
    def process(self, uuids: List[str], data: LibraryDataGenerationPipelineContext) -> None:
        df = data.loaded_audio_info_cache
        mask = df[Header.UUID.value].isin(uuids)
        to_process = df[mask]

        if to_process.empty:
            self._logger.warning("No tracks found to process for energy refill.", separator=self._SEPARATOR)
            return

        processed = self._log_and_handle_exception(
            "Energy calculator failed",
            data.energy_calculator.calculate_ratings_for_df,
            to_process,
            Header.Energy_Level
        )

        if processed is not None:
            processed_energies = processed.set_index(Header.UUID.value)[Header.Energy_Level.value]
            df.set_index(Header.UUID.value, inplace=True)
            df.loc[processed_energies.index, Header.Energy_Level.value] = processed_energies
            data.loaded_audio_info_cache = df.reset_index()
