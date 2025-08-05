from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext


class EnergyLevelCalculatorPipe(IPipe):
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = "BuildCSV.EnergyLevelCalculatorPipe"

    def flow(self, context: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        if context.generated_audio_info.empty:
            self._logger.debug("Skipping energy level calculation, no new audio info.", separator=self._separator)
            return context

        self._logger.info("Applying final energy level calculation...", separator=self._separator)
        context.generated_audio_info = context.energy_calculator.calculate_ratings_for_df(
            context.generated_audio_info, Header.Energy_Level
        )
        return context
