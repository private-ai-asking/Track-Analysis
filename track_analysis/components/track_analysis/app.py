import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Callable, TypeVar, Any

from sentence_transformers import SentenceTransformer
from viztracer import VizTracer

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.cli_framework import CommandLineInterface
from track_analysis.components.md_common_python.py_common.command_handling import CommandHelper
from track_analysis.components.md_common_python.py_common.component_registration import ComponentRegistration
from track_analysis.components.md_common_python.py_common.handlers import FileHandler
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.testing import TestCoordinator
from track_analysis.components.md_common_python.py_common.testing.test_coordinator import TestConfiguration
from track_analysis.components.md_common_python.py_common.time_handling import TimeUtils
from track_analysis.components.md_common_python.py_common.user_input.user_input_helper import UserInputHelper
from track_analysis.components.md_common_python.py_common.utils import SimilarityScorer
from track_analysis.components.md_common_python.py_common.utils.string_utils import StringUtils
from track_analysis.components.track_analysis.constants import ROOT_MUSIC_LIBRARY, OUTPUT_DIRECTORY, \
    DATA_DIRECTORY, BENCHMARK_DIRECTORY, DELETE_FINAL_DATA_BEFORE_START, CACHE_DIRECTORY, CLEAR_CACHE, \
    DOWNLOAD_CSV_FILE, TEST_SAMPLE_SIZE, PROFILE_DATA_LOADING, EMBED_BATCH_SIZE, NUM_WORKERS_CPU_HEAVY, \
    MAX_NEW_TRACKS_PER_RUN
from track_analysis.components.track_analysis.features.audio_calculator import AudioCalculator
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler
from track_analysis.components.track_analysis.features.data_generation.pipeline.build_csv_pipeline import \
    BuildLibraryDataCSVPipeline
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.key_extraction.core.definitions.definition_templates import \
    TemplateMode
from track_analysis.components.track_analysis.features.key_extraction.profile_generation.profile_generator import \
    ProfileGenerator
from track_analysis.components.track_analysis.features.scrobbling.embedding.default_candidate_retriever import \
    DefaultCandidateRetriever
from track_analysis.components.track_analysis.features.scrobbling.embedding.embedding_searcher import EmbeddingSearcher
from track_analysis.components.track_analysis.features.scrobbling.get_unmatched_library_tracks import \
    UnmatchedLibraryTracker
from track_analysis.components.track_analysis.features.scrobbling.scrobble_linker_service import ScrobbleLinkerService
from track_analysis.components.track_analysis.features.scrobbling.uncertain_keys_processor import UncertainKeysProcessor
from track_analysis.components.track_analysis.features.scrobbling.utils.cache_helper import ScrobbleCacheHelper
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_data_loader import ScrobbleDataLoader
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_utility import ScrobbleUtility
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor
from track_analysis.components.track_analysis.features.track_downloading.api.metadata_api import MetadataAPI
from track_analysis.components.track_analysis.features.track_downloading.api.music_download_interface import \
    MusicDownloadInterface
from track_analysis.components.track_analysis.features.track_downloading.api.ytdlp_music_downloader import \
    YTDLPMusicDownloader
from track_analysis.components.track_analysis.features.track_downloading.model.download_model import DownloadModel
from track_analysis.components.track_analysis.features.track_downloading.pipeline.download_pipeline import \
    DownloadPipeline
from track_analysis.components.track_analysis.features.track_downloading.pipeline.download_pipeline_context import \
    DownloadPipelineContext
from track_analysis.components.track_analysis.features.track_downloading.utils.genre_algorithm import GenreAlgorithm
from track_analysis.tests.embedding_test import EmbeddingTest
from track_analysis.tests.key_test import KeyProgressionTest
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
        library_data_path: Path = OUTPUT_DIRECTORY.joinpath("data.csv")
        scrobble_data_path: Path = DATA_DIRECTORY.joinpath("scrobbles.csv")
        # scrobble_data_path: Path = DATA_DIRECTORY.joinpath("scrobbles_test.csv")
        # scrobble_data_path: Path = DATA_DIRECTORY / "training" / "gold_standard.csv"
        cache_path: Path = CACHE_DIRECTORY.joinpath("scrobble_cache.json")
        gold_standard_csv_path: Path = DATA_DIRECTORY.joinpath("training", "gold_standard.csv")
        manual_override_path: Path = CACHE_DIRECTORY.joinpath("manual_override.json")
        music_track_download_dir: Path = OUTPUT_DIRECTORY / "Music Track Downloads"

        self._embedder: SentenceTransformer = SentenceTransformer(model_name_or_path=str(DATA_DIRECTORY / "__internal__" / "all-MiniLM-l12-v2-embed"), device="cuda")

        keys_path: Path = DATA_DIRECTORY.joinpath("__internal__", "lib_keys.pkl")

        embed_weights={ 'title': 0.35, 'artist': 0.4, 'album': 0.25 }

        self._string_utils: StringUtils = StringUtils(logger)
        self._user_input_helper: UserInputHelper = UserInputHelper(logger)
        self._tag_extractor: TagExtractor = TagExtractor(logger)
        self._file_handler: FileHandler = FileHandler()
        self._audio_file_handler: AudioFileHandler = AudioFileHandler(logger, num_workers=NUM_WORKERS_CPU_HEAVY)
        self._audio_calculator: AudioCalculator = AudioCalculator(logger)
        self._time_utils: TimeUtils = TimeUtils()
        self._registration: ComponentRegistration = ComponentRegistration(logger, port=50000, component_port=50002)
        self._downloader: MusicDownloadInterface = YTDLPMusicDownloader(logger, music_track_download_dir)
        self._genre_algorithm: GenreAlgorithm = GenreAlgorithm(logger)
        self._metadata_api: MetadataAPI = MetadataAPI(logger, self._genre_algorithm)
        self._command_helper: CommandHelper = CommandHelper(logger, "CommandHelper")
        self._profile_creator: ProfileGenerator = ProfileGenerator(logger, template_profile_normalized_to=100, num_workers=NUM_WORKERS_CPU_HEAVY)

        self._download_pipeline: DownloadPipeline = DownloadPipeline(
            logger,
            self._downloader,
            self._command_helper
        )
        self._download_pipeline.build_pipeline()

        self._combo_key: str = "||"

        if CLEAR_CACHE:
            cache_path.unlink(missing_ok=True)

        cache_builder: CacheBuilder = CacheBuilder(logger, cache_path, tree_separator=self._combo_key)
        scrobble_utils: ScrobbleUtility = ScrobbleUtility(logger, self._embedder, embed_weights, join_key=self._combo_key, embed_batch_size=EMBED_BATCH_SIZE)
        token_accept_threshold: float = 70
        scorer = SimilarityScorer(
            embed_weights,
            logger=logger,
            threshold=token_accept_threshold
        )

        self._scrobble_data_loader: ScrobbleDataLoader = ScrobbleDataLoader(logger, library_data_path, scrobble_data_path, self._string_utils, scrobble_utils, DATA_DIRECTORY / "__internal__", keys_path)

        def _load():
            self._scrobble_data_loader.load(TEST_SAMPLE_SIZE)

        if PROFILE_DATA_LOADING:
            _run_with_profiling(_load, "Data Loading")
        else: _load()

        scrobble_cache_helper: ScrobbleCacheHelper = ScrobbleCacheHelper(logger, self._scrobble_data_loader, cache_builder)
        embedding_searcher: EmbeddingSearcher = EmbeddingSearcher(logger, top_k=5, loader=self._scrobble_data_loader, utility=scrobble_utils, candidate_retriever=DefaultCandidateRetriever(
            logger=logger,
            loader=self._scrobble_data_loader,
            token_similarity_scorer=scorer
        ))

        self._uncertain_keys_processor: UncertainKeysProcessor = UncertainKeysProcessor(logger, embedding_searcher, scrobble_utils, self._scrobble_data_loader, manual_override_path)
        self._unmatch_util: UnmatchedLibraryTracker = UnmatchedLibraryTracker(logger, self._scrobble_data_loader, cache_path)

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
            embedding_searcher=embedding_searcher,
            token_accept_threshold=token_accept_threshold,
            scorer=scorer
        )

        registration_test: RegistrationTest = RegistrationTest(logger, self._registration)
        embedding_test: EmbeddingTest = EmbeddingTest(logger, embedder=self._embedder, keys_path=keys_path, data_loader=self._scrobble_data_loader)

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
            )
        ]

        self._test_coordinator: TestCoordinator = TestCoordinator(logger, tests)

        self._logger = logger

    def run(self):
        cmd: CommandLineInterface = CommandLineInterface(self._logger, exit_command=self._exit)
        cmd.add_command(["launch_tests", "lt"], "Hops into the test framework CLI.", self._test_coordinator.start_test_cli)
        cmd.add_command(["make_csv", "mc"], "Builds a CSV for the library data. Fills in any missing values for existing entries.", self._make_csv, arguments=[False])
        cmd.add_command(["make_csv-p", "mc-p"], "Builds a CSV for the library data. Fills in any missing values for existing entries. Also profiles the function.", self._make_csv, arguments=[True])
        cmd.add_command(["generate_embeddings", "ge"], "Generates embeddings for the library.", self._generate_embeddings)
        cmd.add_command(["test_params", "tp"], "Tests various parameter combinations for the algorithm.", self._scrobble_linker.test_parameters)
        cmd.add_command(["test_keys_extraction", "tke"], "Tests the key extraction algorithm.", self._test_keys, arguments=[False])
        cmd.add_command(["test_keys_extraction-p", "tke-p"], "Tests the key extraction algorithm and profiles.", self._test_keys, arguments=[True])
        cmd.add_command(["compute_profiles", "cp"], "Computes profiles based on a corpus.", self._profile_creation_test)
        cmd.add_command(["process_uncertain", "pu"], "Processes the uncertain keys interactively.", self._uncertain_keys_processor.process)
        cmd.add_command(["print_unmatched", "pru"], "Prints the library entries whose UUIDs don't have an associated cached scrobble.", self._unmatch_util.print_unmatched_tracks)
        cmd.add_command(["build_cache", "bc"], "Builds the cache for the library for n samples.", self._build_cache, arguments=[False])
        cmd.add_command(["build_cache-profile", "bc-p"], "Builds the cache for the library for n samples and profiles the performance.", self._build_cache, arguments=[True])
        cmd.add_command(["link_scrobbles", "ls"], "Links the scrobbles to the library data.", self._link_scrobbles, arguments=[False])
        cmd.add_command(["link_scrobbles-profile", "ls-p"], "Links the scrobbles to the library data but profiles the performance also.", self._link_scrobbles, arguments=[True])
        cmd.add_command(["download_and_md", "damd"], "Combines downloading and setting metadata.", self._download_and_assign_metadata)
        cmd.add_command(["add_album_to_downloads", "aatd"], "Adds an album to the downloads.csv file.", self._metadata_api.add_album_to_downloads)

        # noinspection PyBroadException
        try:
            cmd.start_listen_loop()
        except Exception as _:
            tb = traceback.format_exc()
            self._logger.critical(
                f"Something went terribly wrong, causing the application to nearly crash. Restarting.\n{tb}"
            )
            self.run()

    def _profile_creation_test(self):
        corpus_path = Path(r"X:\Track Analysis\data\training\corpus.csv")

        def __run():
            self._profile_creator.generate_profile(corpus_path)

        __run()

    def _download_and_assign_metadata(self):
        # Run the download pipeline
        data: DownloadPipelineContext = self._download_pipeline.flow(
            DownloadPipelineContext(download_csv_path=DOWNLOAD_CSV_FILE)
        )
        download_files: List[DownloadModel] = data.downloaded_tracks

        # Choose a sensible pool size (e.g. cap at 32 threads for I/O-bound work)
        max_workers = min(32, len(download_files) or 1)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all metadata-population tasks
            future_to_track = {
                executor.submit(
                    self._metadata_api.populate_metadata_from_musicbrainz_for_file,
                    track
                ): track
                for track in download_files
            }

            # As each task completes, handle its result or exception
            for future in as_completed(future_to_track):
                track = future_to_track[future]
                try:
                    future.result()
                except Exception as e:
                    self._logger.error(
                        f"Error populating metadata for {track}: {e}"
                    )

    def _test_keys(self, profiling: bool = False) -> None:
        track_path: Path = Path(r"W:\media\music\[02] organized\[01] hq\Reggae\Nas\Distant Relatives\11 Nas & Damian Marley - Patience.flac")
        # track_path: Path = Path(r"W:\media\music\[02] organized\[01] hq\Classical\Claude Debussy\Classical Best\31 Danse sacrée et progane - Sacred Dance.flac")
        # track_path: Path = Path(r"W:\media\music\[02] organized\[02] lq\CCM\Champion\08 - Beckah Shae - Me and My God.flac")
        tester: KeyProgressionTest = KeyProgressionTest(self._logger, tone_modulation_penalty=12.0, mode_modulation_penalty=None, visualize=False, template_mode=TemplateMode.KS_T_REVISED, segment_beat_level=4)

        def __test():
            tester.test(file_path=track_path)

        if profiling: _run_with_profiling(__test, category="Key Extraction")
        else: __test()

    def _exit(self):
        self._registration.shutdown_component()
        self._logger.save()
        exit()

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

    def _make_csv(self, profile: bool = False) -> None:
        output_path: Path = OUTPUT_DIRECTORY.joinpath("data.csv")

        pipeline_context = LibraryDataGenerationPipelineContext(
            source_dir=ROOT_MUSIC_LIBRARY,
            main_data_output_file_path=output_path,
            use_threads=True,
            max_new_tracks_per_run=MAX_NEW_TRACKS_PER_RUN,
            missing_headers_to_fill=[],
            headers_to_refill=[]
        )

        # output_path.unlink(missing_ok=True)

        pipeline = BuildLibraryDataCSVPipeline(self._logger, self._file_handler, self._tag_extractor, self._audio_file_handler, self._audio_calculator, self._string_utils, num_workers=NUM_WORKERS_CPU_HEAVY)

        def __run():
            pipeline.build_pipeline()
            pipeline.flow(pipeline_context)

        if profile:
            _run_with_profiling(__run, "Building Library CSV")
        else: __run()

        self._logger.info("CSV has been successfully created.")
