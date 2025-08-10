import traceback
from pathlib import Path
from typing import List

from track_analysis.components.md_common_python.py_common.cli_framework import CommandLineInterface
from track_analysis.components.md_common_python.py_common.command_handling import CommandHelper
from track_analysis.components.md_common_python.py_common.handlers import FileHandler
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.testing import TestCoordinator
from track_analysis.components.md_common_python.py_common.testing.test_coordinator import TestConfiguration
from track_analysis.components.md_common_python.py_common.utils.string_utils import StringUtils
from track_analysis.components.track_analysis.command_execution.command_factory import CommandFactory
from track_analysis.components.track_analysis.command_execution.command_line_initiator import CommandLineInitiator
from track_analysis.components.track_analysis.features.data_generation.factory.csv_pipeline_factory import \
    CsvPipelineFactory
from track_analysis.components.track_analysis.features.scrobble_linking.factory.scrobble_factory import ScrobbleFactory
from track_analysis.components.track_analysis.features.scrobble_linking.get_unmatched_library_tracks import \
    UnmatchedLibraryTracker
from track_analysis.components.track_analysis.features.scrobble_linking.processor.uncertain_keys_processor import \
    UncertainKeysProcessor
from track_analysis.components.track_analysis.features.scrobble_linking.scrobble_linker_service import \
    ScrobbleLinkerService
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor
from track_analysis.components.track_analysis.features.track_downloading.api.metadata_api import MetadataAPI
from track_analysis.components.track_analysis.features.track_downloading.factory.download_pipeline_factory import \
    DownloadPipelineFactory
from track_analysis.components.track_analysis.features.track_downloading.utils.genre_algorithm import GenreAlgorithm
from track_analysis.components.track_analysis.library.configuration.model.configuration import \
    TrackAnalysisConfigurationModel
from track_analysis.components.track_analysis.shared.caching.max_rate_cache import \
    MaxRateCache
from track_analysis.components.track_analysis.shared_objects import MEMORY
from track_analysis.tests.energy_calculation_test import EnergyCalculationTest


class App:
    def __init__(self, logger: HoornLogger, configuration: TrackAnalysisConfigurationModel):
        MEMORY.instantiate(configuration.paths.expensive_cache_dir / "track_analysis_cache.h5")
        MEMORY.set_logger(logger)

        self._logger = logger

        configuration: TrackAnalysisConfigurationModel = configuration
        cache_path: Path = configuration.paths.scrobble_cache

        string_utils: StringUtils = StringUtils(logger)
        tag_extractor: TagExtractor = TagExtractor(logger)
        file_handler: FileHandler = FileHandler()

        self._max_rate_cache: MaxRateCache = MaxRateCache(configuration.paths.max_rate_cache)
        genre_algorithm: GenreAlgorithm = GenreAlgorithm(logger)
        metadata_api: MetadataAPI = MetadataAPI(logger, genre_algorithm, configuration)
        command_helper: CommandHelper = CommandHelper(logger, "CommandHelper")

        scrobble_factory: ScrobbleFactory = ScrobbleFactory(logger, string_utils, configuration)

        if configuration.development.clear_cache:
            cache_path.unlink(missing_ok=True)

        scrobble_linker: ScrobbleLinkerService = scrobble_factory.create_scrobble_linker_service()
        uncertain_keys_processor: UncertainKeysProcessor = scrobble_factory.create_uncertain_keys_processor()
        unmatch_util: UnmatchedLibraryTracker = scrobble_factory.create_unmatched_library_tracker()

        energy_test: EnergyCalculationTest = EnergyCalculationTest(logger, track_analysis_config=configuration)

        tests: List[TestConfiguration] = [
            TestConfiguration(
                associated_test=energy_test,
                keyword_arguments=[],
                command_description="Tests the energy calculation.",
                command_keys=["test_energy_calculation", "tec"]
            )
        ]

        test_coordinator: TestCoordinator = TestCoordinator(logger, tests)
        self._cmd: CommandLineInterface = CommandLineInterface(self._logger, exit_command=self._exit)

        download_pipeline_factory = DownloadPipelineFactory(
            logger,
            command_helper,
            configuration
        )

        csv_pipeline_factory = CsvPipelineFactory(
            logger,
            file_handler,
            string_utils,
            tag_extractor,
            self._max_rate_cache,
            configuration
        )

        command_factory: CommandFactory = CommandFactory(
            logger,
            metadata_api,
            scrobble_linker,
            download_pipeline_factory.create(),
            test_coordinator,
            csv_pipeline_factory,
            unmatch_util,
            uncertain_keys_processor,
            configuration
        )

        self._command_line_initiator: CommandLineInitiator = CommandLineInitiator(logger, self._cmd, command_factory, configuration)

    def run(self):
        self._command_line_initiator.initialize()

        # noinspection PyBroadException
        try:
            self._cmd.start_listen_loop()
        except Exception as _:
            tb = traceback.format_exc()
            self._logger.critical(
                f"Something went terribly wrong, causing the application to nearly crash. Restarting.\n{tb}"
            )
            self.run()

    def on_exit(self) -> None:
        self._exit()

    def _exit(self):
        MEMORY.close()
        self._max_rate_cache.save()
        self._logger.save()
        exit()
