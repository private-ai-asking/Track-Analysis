from pathlib import Path
from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns.command.command_interface import CommandInterface, T, \
    P
from track_analysis.components.track_analysis.command_execution.command_execution_model import CommandExecutionModel
from track_analysis.components.track_analysis.features.scrobble_linking.scrobble_linker_service import \
    ScrobbleLinkerService
from track_analysis.components.track_analysis.library.configuration.model.configuration import \
    TrackAnalysisConfigurationModel


class LinkScrobblesCommand(CommandExecutionModel):
    def __init__(self, logger: HoornLogger, scrobble_linker: ScrobbleLinkerService, configuration: TrackAnalysisConfigurationModel):
        super().__init__(logger)
        self._scrobble_linker = scrobble_linker
        self._configuration = configuration

    @property
    def default_command_keys(self) -> List[str]:
        return ["link_scrobbles", "ls"]

    @property
    def command_description(self) -> str:
        return "Links the scrobble to the library data."

    def execute(self, arguments: T) -> P:
        output_path: Path = self._configuration.paths.enriched_scrobble_data

        if self._configuration.development.delete_final_data_before_start:
            output_path.unlink(missing_ok=True)

        enriched = self._scrobble_linker.link_scrobbles()
        enriched.to_csv(output_path, index=False)
        return enriched

