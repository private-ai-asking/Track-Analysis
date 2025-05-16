import traceback

from datetime import datetime
from pathlib import Path
from typing import List, Callable, TypeVar, Any

from sentence_transformers import SentenceTransformer
from viztracer import VizTracer

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.cli_framework import CommandLineInterface
from track_analysis.components.md_common_python.py_common.component_registration import ComponentRegistration
from track_analysis.components.md_common_python.py_common.handlers import FileHandler
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.testing import TestCoordinator
from track_analysis.components.md_common_python.py_common.testing.test_coordinator import TestConfiguration
from track_analysis.components.md_common_python.py_common.time_handling import TimeUtils
from track_analysis.components.md_common_python.py_common.user_input.user_input_helper import UserInputHelper
from track_analysis.components.md_common_python.py_common.utils.string_utils import StringUtils
from track_analysis.components.track_analysis.constants import ROOT_MUSIC_LIBRARY, OUTPUT_DIRECTORY, \
    DATA_DIRECTORY, BENCHMARK_DIRECTORY, DELETE_FINAL_DATA_BEFORE_START, CACHE_DIRECTORY, CLEAR_CACHE
from track_analysis.components.track_analysis.features.audio_calculator import AudioCalculator
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler
from track_analysis.components.track_analysis.features.data_generation.data_generator import DataGenerator
from track_analysis.components.track_analysis.features.scrobbling.cache_helper import ScrobbleCacheHelper
from track_analysis.components.track_analysis.features.scrobbling.embedding_searcher import EmbeddingSearcher
from track_analysis.components.track_analysis.features.scrobbling.scrobble_data_loader import ScrobbleDataLoader
from track_analysis.components.track_analysis.features.scrobbling.scrobble_linker_service import ScrobbleLinkerService
from track_analysis.components.track_analysis.features.scrobbling.scrobble_utility import ScrobbleUtility
from track_analysis.components.track_analysis.features.scrobbling.uncertain_keys_processor import UncertainKeysProcessor
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.build_csv_pipeline import BuildCSVPipeline
from track_analysis.components.track_analysis.features.data_generation.pipeline.locate_paths_pipeline import LocatePathsPipeline
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import PipelineContextModel
from track_analysis.tests.embedding_test import EmbeddingTest
from track_analysis.tests.extract_tags_test import ExtractTagsTest
from track_analysis.tests.registration_test import RegistrationTest

T = TypeVar("T")


def _run_with_profiling(func: Callable[[], T], category: str) -> T:
    """
    Runs the given function under VizTracer profiling and writes a benchmark file.

    :param func: A no-argument function to execute and profile.
    :param category: The profiling category (used to name the subdirectory).
    :return: The return value of the function.
    """
    # Prepare timestamp and paths
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d-%H_%M_%S")
    benchmark_dir = BENCHMARK_DIRECTORY.joinpath(category)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = benchmark_dir.joinpath(f"{timestamp}-benchmark.json")

    # Run with VizTracer
    with VizTracer(
            output_file=str(benchmark_path),
            min_duration=1,
            ignore_c_function=True,
            ignore_frozen=True
    ) as _:
        result = func()

    return result


class App:
    def __init__(self, logger: HoornLogger):
        self._embedder: SentenceTransformer = SentenceTransformer(model_name_or_path=str(DATA_DIRECTORY / "__internal__" / "all-MiniLM-l12-v2-embed"), device="cuda")
        embedding_searcher: EmbeddingSearcher = EmbeddingSearcher(logger, top_k=5)

        keys_path: Path = DATA_DIRECTORY.joinpath("__internal__", "lib_keys.pkl")

        embed_weights={ 'title': 0.35, 'artist': 0.4, 'album': 0.25 }

        self._string_utils: StringUtils = StringUtils(logger)
        self._user_input_helper: UserInputHelper = UserInputHelper(logger)
        self._tag_extractor: TagExtractor = TagExtractor(logger)
        self._file_handler: FileHandler = FileHandler()
        self._audio_file_handler: AudioFileHandler = AudioFileHandler(logger)
        self._audio_calculator: AudioCalculator = AudioCalculator(logger)
        self._time_utils: TimeUtils = TimeUtils()
        self._registration: ComponentRegistration = ComponentRegistration(logger, port=50000, component_port=50002)
        self._combo_key: str = "||"

        library_data_path: Path = OUTPUT_DIRECTORY.joinpath("data.csv")
        scrobble_data_path: Path = DATA_DIRECTORY.joinpath("scrobbles.csv")
        # scrobble_data_path: Path = DATA_DIRECTORY.joinpath("scrobbles_test.csv")
        # scrobble_data_path: Path = DATA_DIRECTORY / "training" / "gold_standard.csv"
        cache_path: Path = CACHE_DIRECTORY.joinpath("scrobble_cache.json")
        gold_standard_csv_path: Path = DATA_DIRECTORY.joinpath("training", "gold_standard.csv")
        manual_override_path: Path = CACHE_DIRECTORY.joinpath("manual_override.json")

        if CLEAR_CACHE:
            cache_path.unlink(missing_ok=True)

        cache_builder: CacheBuilder = CacheBuilder(logger, cache_path, tree_separator=self._combo_key)
        scrobble_utils: ScrobbleUtility = ScrobbleUtility(logger, self._embedder, embed_weights, join_key=self._combo_key, embed_batch_size=248)

        self._scrobble_data_loader: ScrobbleDataLoader = ScrobbleDataLoader(logger, library_data_path, scrobble_data_path, self._string_utils, scrobble_utils, DATA_DIRECTORY / "__internal__", keys_path)
        scrobble_cache_helper: ScrobbleCacheHelper = ScrobbleCacheHelper(logger, self._scrobble_data_loader, cache_builder)

        self._uncertain_keys_processor: UncertainKeysProcessor = UncertainKeysProcessor(logger, embedding_searcher, scrobble_utils, self._scrobble_data_loader, manual_override_path)

        self._scrobble_linker: ScrobbleLinkerService = ScrobbleLinkerService(
            logger,
            data_loader=self._scrobble_data_loader,
            string_utils=self._string_utils,
            embedder=self._embedder,
            keys_path=keys_path,
            combo_key=self._combo_key,
            scrobble_utils=scrobble_utils,
            cache_builder=cache_builder,
            gold_standard_csv_path=gold_standard_csv_path,
            embed_weights=embed_weights,
            cache_helper=scrobble_cache_helper,
            manual_override_path=manual_override_path,
            embedding_searcher=embedding_searcher
        )

        registration_test: RegistrationTest = RegistrationTest(logger, self._registration)
        embedding_test: EmbeddingTest = EmbeddingTest(logger, embedder=self._embedder, keys_path=keys_path, data_loader=self._scrobble_data_loader)
        extract_tags_test: ExtractTagsTest = ExtractTagsTest(logger, self._user_input_helper, self._tag_extractor)

        tests: List[TestConfiguration] = [
            TestConfiguration(
                associated_test=registration_test,
                keyword_arguments=[],
                command_description="Tests the registration functionality.",
                command_keys=["test_registration", "tr"]
            ),
            TestConfiguration(
                associated_test=embedding_test,
                keyword_arguments=[10],
                command_description="Tests the embedding similarity matcher.",
                command_keys=["test_embeddings", "te"]
            ),
            TestConfiguration(
                associated_test=extract_tags_test,
                keyword_arguments=[],
                command_description="Debugs the extract tags function.",
                command_keys=["extract_tags_debug", "etd"]
            ),
        ]

        self._test_coordinator: TestCoordinator = TestCoordinator(logger, tests)

        self._logger = logger

    def run(self):
        cmd: CommandLineInterface = CommandLineInterface(self._logger, exit_command=self._exit)
        cmd.add_command(["launch_tests", "lt"], "Hops into the test framework CLI.", self._test_coordinator.start_test_cli)
        cmd.add_command(["make_csv", "mc"], "Makes a CSV file from the extracted metadata.", self._make_csv)
        cmd.add_command(["add_path_to_metadata", "apm"], "Adds the path of a file to the metadata.", self._add_path_to_metadata)
        cmd.add_command(["generate_new_data", "gnd"], "Fills in the newly added header(s) since last cache update.", self._generate_new_data)
        cmd.add_command(["generate_embeddings", "ge"], "Generates embeddings for the library.", self._generate_embeddings)
        cmd.add_command(["test_params", "tp"], "Tests various parameter combinations for the algorithm.", self._scrobble_linker.test_parameters)
        cmd.add_command(["process_uncertain", "pu"], "Processes the uncertain keys interactively.", self._uncertain_keys_processor.process)
        cmd.add_command(["build_cache", "bc"], "Builds the cache for the library for n samples.", self._build_cache, arguments=[False])
        cmd.add_command(["build_cache-profile", "bc-p"], "Builds the cache for the library for n samples and profiles the performance.", self._build_cache, arguments=[True])
        cmd.add_command(["link_scrobbles", "ls"], "Links the scrobbles to the library data.", self._link_scrobbles, arguments=[False])
        cmd.add_command(["link_scrobbles-profile", "ls-p"], "Links the scrobbles to the library data but profiles the performance also.", self._link_scrobbles, arguments=[True])

        # noinspection PyBroadException
        try:
            cmd.start_listen_loop()
        except Exception as _:
            tb = traceback.format_exc()
            self._logger.error(
                f"Something went terribly wrong, causing the application to nearly crash. Restarting.\n{tb}"
            )
            self.run()

    def _exit(self):
        self._registration.shutdown_component()
        self._logger.save()
        exit()

    def _generate_new_data(self):
        data_generator: DataGenerator = DataGenerator(self._logger, self._audio_file_handler, self._audio_calculator, self._time_utils)
        data_generator.generate_data([Header.True_Peak], batch_size=32)

    def _generate_embeddings(self):
        self._scrobble_linker.build_embeddings_for_library()

    def _build_cache(self, profiling: bool = False) -> None:
        if not profiling:
            self._scrobble_linker.build_cache()
            return

        _run_with_profiling(self._scrobble_linker.build_cache, "Embedding Construction")

    def _link_scrobbles(self, profiling: bool = False) -> None:
        output_path: Path = OUTPUT_DIRECTORY.joinpath("enriched_scrobbles.csv")

        if DELETE_FINAL_DATA_BEFORE_START:
            output_path.unlink(missing_ok=True)

        def _perform_linking() -> Any:
            # Encapsulate the core logic
            enriched = self._scrobble_linker.link_scrobbles()
            enriched.to_csv(output_path, index=False)
            return enriched

        if profiling:
            _run_with_profiling(_perform_linking, "Scrobble Matching")
            return

        # Regular execution
        _perform_linking()

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
