import cProfile
import pstats
from pathlib import Path
from pstats import Stats

import speedscope

from track_analysis.components.md_common_python.py_common.cli_framework import CommandLineInterface
from track_analysis.components.md_common_python.py_common.component_registration import ComponentRegistration
from track_analysis.components.md_common_python.py_common.handlers import FileHandler
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.time_handling import TimeUtils
from track_analysis.components.md_common_python.py_common.user_input.user_input_helper import UserInputHelper
from track_analysis.components.md_common_python.py_common.utils.string_utils import StringUtils
from track_analysis.components.track_analysis.constants import ROOT_MUSIC_LIBRARY, OUTPUT_DIRECTORY, \
    MINIMUM_FUZZY_CONFIDENCE, DATA_DIRECTORY
from track_analysis.components.track_analysis.features.audio_calculator import AudioCalculator
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler
from track_analysis.components.track_analysis.features.data_generation.data_generator import DataGenerator
from track_analysis.components.track_analysis.features.scrobbling.scrobble_linker_service import ScrobbleLinkerService
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor
from track_analysis.components.track_analysis.model.audio_info import AudioInfo
from track_analysis.components.track_analysis.model.header import Header
from track_analysis.components.track_analysis.pipeline.build_csv_pipeline import BuildCSVPipeline

from track_analysis.components.track_analysis.pipeline.locate_paths_pipeline import LocatePathsPipeline
from track_analysis.components.track_analysis.pipeline.pipeline_context import PipelineContextModel


class App:
    def __init__(self, logger: HoornLogger):
        self._string_utils: StringUtils = StringUtils(logger)
        self._user_input_helper: UserInputHelper = UserInputHelper(logger)
        self._tag_extractor: TagExtractor = TagExtractor(logger)
        self._file_handler: FileHandler = FileHandler()
        self._audio_file_handler: AudioFileHandler = AudioFileHandler(logger)
        self._audio_calculator: AudioCalculator = AudioCalculator(logger)
        self._time_utils: TimeUtils = TimeUtils()
        self._registration: ComponentRegistration = ComponentRegistration(logger, port=50000, component_port=50002)
        self._scrobble_linker: ScrobbleLinkerService = ScrobbleLinkerService(
            logger,
            library_data_path=OUTPUT_DIRECTORY.joinpath("data.csv"),
            scrobble_data_path=DATA_DIRECTORY.joinpath("scrobbles_test.csv"),
            string_utils=self._string_utils
        )
        self._logger = logger

    def run(self):
        cmd: CommandLineInterface = CommandLineInterface(self._logger, exit_command=self._exit)
        cmd.add_command(["test_registration", "tr"], "Tests the registration functionality.", self._test_registration)
        cmd.add_command(["extract_tags_debug", "etd"], "Debugs the extract tags function.", self._debug_extract_tags)
        cmd.add_command(["make_csv", "mc"], "Makes a CSV file from the extracted metadata.", self._make_csv)
        cmd.add_command(["add_path_to_metadata", "apm"], "Adds the path of a file to the metadata.", self._add_path_to_metadata)
        cmd.add_command(["generate_new_data", "gnd"], "Fills in the newly added header(s) since last cache update.", self._generate_new_data)
        cmd.add_command(["link_scrobbles", "ls"], "Links the scrobbles to the library data.", self._link_scrobbles, arguments=[False])
        cmd.add_command(["link_scrobbles-profile", "ls-p"], "Links the scrobbles to the library data but profiles the performance also.", self._link_scrobbles, arguments=[True])
        cmd.start_listen_loop()

    def _exit(self):
        self._registration.shutdown_component()
        exit()

    def _generate_new_data(self):
        data_generator: DataGenerator = DataGenerator(self._logger, self._audio_file_handler, self._audio_calculator, self._time_utils)
        data_generator.generate_data([Header.True_Peak], batch_size=32)

    def _link_scrobbles(self, profiling: bool=False) -> None:
        output_path: Path = OUTPUT_DIRECTORY.joinpath("enriched_scrobbles.csv")

        if not profiling:
            self._scrobble_linker.link_scrobbles(output_path, threshold=MINIMUM_FUZZY_CONFIDENCE)
            return

        # with cProfile.Profile() as profile:
        #     self._scrobble_linker.link_scrobbles(output_path, threshold=MINIMUM_FUZZY_CONFIDENCE)
        #
        # results: Stats = Stats(profile)
        # results.sort_stats(pstats.SortKey.TIME)
        # results.print_stats()

        with speedscope.track(OUTPUT_DIRECTORY.joinpath("benchmarks/speedscope.json")):
            self._scrobble_linker.link_scrobbles(output_path, threshold=MINIMUM_FUZZY_CONFIDENCE)

    def _test_registration(self):
        registration_path: Path = Path("X:\\Track Analysis\\track_analysis\components\\track_analysis\\registration.json")
        signature_path: Path = Path("X:\\Track Analysis\\track_analysis\components\\track_analysis\\component_signature.json")

        self._registration.register_component(registration_path, signature_path)

    def _debug_extract_tags(self):
        def _always_true_validator(_: str) -> (bool, str):
            return True, ""

        # W:\media\music\[02] organized\[01] hq\Classical\Ludovico Einaudi\Elegy For The Arctic\01 Elegy for the Arctic.flac
        # W:\media\music\[02] organized\[02] lq\CCM\Champion\15 - Beckah Shae - Incorruptible (David Thulin remix).flac
        file_to_check: str = self._user_input_helper.get_user_input("Please enter the path to the audio file you want to extract:", str, validator_func=_always_true_validator)

        result: AudioInfo =self._tag_extractor.extract(Path(file_to_check))

        for metadata_item in result.metadata:
            self._logger.info(f"{metadata_item.header} - {metadata_item.description}: {metadata_item.value}")

    def _add_path_to_metadata(self):
        pipeline_context = PipelineContextModel(
            source_dir=ROOT_MUSIC_LIBRARY,
            main_data_output_file_path=OUTPUT_DIRECTORY.joinpath("data.csv"),
        )

        pipeline = LocatePathsPipeline(self._logger, self._file_handler, self._tag_extractor)
        pipeline.build_pipeline()
        pipeline.flow(pipeline_context)

        self._logger.info("Paths have been successfully matched.")

    def _make_csv(self):
        pipeline_context = PipelineContextModel(
            source_dir=ROOT_MUSIC_LIBRARY,
            main_data_output_file_path=OUTPUT_DIRECTORY.joinpath("data.csv")
        )

        pipeline = BuildCSVPipeline(self._logger, self._file_handler, self._tag_extractor, self._audio_file_handler, self._audio_calculator)
        pipeline.build_pipeline()
        pipeline.flow(pipeline_context)

        self._logger.info("CSV has been successfully created.")
