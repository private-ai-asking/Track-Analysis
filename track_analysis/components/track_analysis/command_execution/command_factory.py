from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.testing import TestCoordinator
from track_analysis.components.track_analysis.command_execution.command_execution_model import CommandExecutionModel
from track_analysis.components.track_analysis.command_execution.commands.add_album_to_downloads import \
    AddAlbumToDownloadsCommand
from track_analysis.components.track_analysis.command_execution.commands.build_scrobble_cache import \
    BuildScrobbleCacheCommand
from track_analysis.components.track_analysis.command_execution.commands.download_and_assign_metadata import \
    DownloadAndAssignMetadataCommand
from track_analysis.components.track_analysis.command_execution.commands.generate_embeddings import GenerateEmbeddingsCommand
from track_analysis.components.track_analysis.command_execution.commands.launch_tests import LaunchTestsCommand
from track_analysis.components.track_analysis.command_execution.commands.link_scrobbles import LinkScrobblesCommand
from track_analysis.components.track_analysis.command_execution.commands.log_hdf5_stats import LogHDF5StatsCommand
from track_analysis.components.track_analysis.command_execution.commands.make_csv import MakeCSVCommand
from track_analysis.components.track_analysis.command_execution.commands.print_unmatched_uuids import \
    PrintUnmatchedUUIDsCommand
from track_analysis.components.track_analysis.command_execution.commands.process_uncertain_keys import \
    ProcessUncertainKeysCommand
from track_analysis.components.track_analysis.command_execution.commands.save_directory_tree import \
    SaveDirectoryTreeCommand
from track_analysis.components.track_analysis.command_execution.commands.test_parameters import TestParametersCommand
from track_analysis.components.track_analysis.features.data_generation.factory.csv_pipeline_factory import \
    CsvPipelineFactory
from track_analysis.components.track_analysis.features.scrobble_linking.get_unmatched_library_tracks import \
    UnmatchedLibraryTracker
from track_analysis.components.track_analysis.features.scrobble_linking.processor.uncertain_keys_processor import \
    UncertainKeysProcessor
from track_analysis.components.track_analysis.features.scrobble_linking.scrobble_linker_service import \
    ScrobbleLinkerService
from track_analysis.components.track_analysis.features.track_downloading.api.metadata_api import MetadataAPI
from track_analysis.components.track_analysis.features.track_downloading.pipeline.download_pipeline import \
    DownloadPipeline
from track_analysis.components.track_analysis.library.configuration.model.configuration import \
    TrackAnalysisConfigurationModel


class CommandFactory:
    def __init__(self,
                 logger: HoornLogger,
                 metadata_api: MetadataAPI,
                 scrobble_linker: ScrobbleLinkerService,
                 download_pipeline: DownloadPipeline,
                 test_coordinator: TestCoordinator,
                 csv_pipeline_factory: CsvPipelineFactory,
                 unmatched_library_tracker: UnmatchedLibraryTracker,
                 uncertain_keys_processor: UncertainKeysProcessor,
                 configuration: TrackAnalysisConfigurationModel):
        self._logger = logger
        self._metadata_api = metadata_api
        self._scrobble_linker = scrobble_linker
        self._download_pipeline = download_pipeline
        self._test_coordinator = test_coordinator
        self._csv_pipeline_factory = csv_pipeline_factory
        self._unmatched_library_tracker = unmatched_library_tracker
        self._uncertain_keys_processor = uncertain_keys_processor

        self._configuration = configuration

    def get_commands(self) -> List[CommandExecutionModel]:
        return [
            AddAlbumToDownloadsCommand(self._logger, self._metadata_api),
            BuildScrobbleCacheCommand(self._logger, self._scrobble_linker),
            DownloadAndAssignMetadataCommand(self._logger, self._download_pipeline, self._metadata_api, self._configuration),
            GenerateEmbeddingsCommand(self._logger, self._scrobble_linker),
            LaunchTestsCommand(self._logger, self._test_coordinator),
            LinkScrobblesCommand(self._logger, self._scrobble_linker, self._configuration),
            LogHDF5StatsCommand(self._logger),
            MakeCSVCommand(self._logger, self._configuration, self._csv_pipeline_factory),
            PrintUnmatchedUUIDsCommand(self._logger, self._unmatched_library_tracker),
            ProcessUncertainKeysCommand(self._logger, self._uncertain_keys_processor),
            SaveDirectoryTreeCommand(self._logger),
            TestParametersCommand(self._logger, self._scrobble_linker)
        ]
