import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Callable, TypeVar, Any

from viztracer import VizTracer

from track_analysis.components.md_common_python.py_common.cli_framework import CommandLineInterface
from track_analysis.components.md_common_python.py_common.command_handling import CommandHelper
from track_analysis.components.md_common_python.py_common.handlers import FileHandler
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.misc import DirectoryTreeConfig, DirectoryTreeGenerator
from track_analysis.components.md_common_python.py_common.testing import TestCoordinator
from track_analysis.components.md_common_python.py_common.testing.test_coordinator import TestConfiguration
from track_analysis.components.md_common_python.py_common.time_handling import TimeUtils
from track_analysis.components.md_common_python.py_common.user_input.user_input_helper import UserInputHelper
from track_analysis.components.md_common_python.py_common.utils.string_utils import StringUtils
from track_analysis.components.track_analysis.features.data_generation.build_csv_pipeline import \
    BuildLibraryDataCSVPipeline, PipelineConfiguration
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.scrobble_linking.factory.scrobble_factory import ScrobbleFactory
from track_analysis.components.track_analysis.features.scrobble_linking.get_unmatched_library_tracks import \
    UnmatchedLibraryTracker
from track_analysis.components.track_analysis.features.scrobble_linking.processor.uncertain_keys_processor import \
    UncertainKeysProcessor
from track_analysis.components.track_analysis.features.scrobble_linking.scrobble_linker_service import \
    ScrobbleLinkerService
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
from track_analysis.components.track_analysis.library.configuration.model.configuration import \
    TrackAnalysisConfigurationModel
# from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.profile_generation.profile_generator import \
#     ProfileGenerator
from track_analysis.components.track_analysis.shared.caching.max_rate_cache import \
    MaxRateCache
from track_analysis.components.track_analysis.shared_objects import MEMORY
from track_analysis.tests.energy_calculation_test import EnergyCalculationTest

T = TypeVar("T")


def _run_with_profiling(func: Callable[[], T], category: str, benchmark_dir: Path) -> T:
    """
    Runs the given function under VizTracer profiling and writes a benchmark file.

    :param func: A no-argument function to execute and profile.
    :param category: The profiling category (used to name the subdirectory).
    :return: The return value of the function.
    """
    # Prepare timestamp and paths
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d-%H_%M_%S")
    benchmark_dir = benchmark_dir.joinpath(category)
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
    def __init__(self, logger: HoornLogger, configuration: TrackAnalysisConfigurationModel):
        MEMORY.instantiate(configuration.paths.expensive_cache_dir / "track_analysis_cache.h5")
        MEMORY.set_logger(logger)

        self._configuration: TrackAnalysisConfigurationModel = configuration
        self._library_data_path: Path = configuration.paths.library_data
        self._mfcc_data_path: Path = configuration.paths.mfcc_data
        cache_path: Path = configuration.paths.scrobble_cache
        music_track_download_dir: Path = configuration.paths.music_track_downloads

        self._string_utils: StringUtils = StringUtils(logger)
        self._user_input_helper: UserInputHelper = UserInputHelper(logger)
        self._tag_extractor: TagExtractor = TagExtractor(logger)
        self._file_handler: FileHandler = FileHandler()

        self._max_rate_cache: MaxRateCache = MaxRateCache(configuration.paths.max_rate_cache)

        self._time_utils: TimeUtils = TimeUtils()
        self._downloader: MusicDownloadInterface = YTDLPMusicDownloader(logger, music_track_download_dir, configuration.paths.cookies_file, configuration.paths.ffmpeg_path)
        self._genre_algorithm: GenreAlgorithm = GenreAlgorithm(logger)
        self._metadata_api: MetadataAPI = MetadataAPI(logger, self._genre_algorithm, configuration)
        self._command_helper: CommandHelper = CommandHelper(logger, "CommandHelper")
        # self._profile_creator: ProfileGenerator = ProfileGenerator(logger, self._audio_file_handler, template_profile_normalized_to=100, num_workers=NUM_WORKERS_CPU_HEAVY-14)
        # self._key_extractor = KeyExtractor(logger, self._audio_file_handler, num_workers=NUM_WORKERS_CPU_HEAVY-14)

        scrobble_factory: ScrobbleFactory = ScrobbleFactory(logger, self._string_utils, configuration)

        self._download_pipeline: DownloadPipeline = DownloadPipeline(
            logger,
            self._downloader,
            self._command_helper,
            configuration.paths.ffmpeg_path,
        )
        self._download_pipeline.build_pipeline()

        if configuration.development.clear_cache:
            cache_path.unlink(missing_ok=True)

        self._scrobble_linker: ScrobbleLinkerService = scrobble_factory.create_scrobble_linker_service()
        self._uncertain_keys_processor: UncertainKeysProcessor = scrobble_factory.create_uncertain_keys_processor()
        self._unmatch_util: UnmatchedLibraryTracker = scrobble_factory.create_unmatched_library_tracker()

        energy_test: EnergyCalculationTest = EnergyCalculationTest(logger, self._library_data_path, mfcc_data_path=self._mfcc_data_path, track_analysis_config=configuration)

        tests: List[TestConfiguration] = [
            TestConfiguration(
                associated_test=energy_test,
                keyword_arguments=[],
                command_description="Tests the energy calculation.",
                command_keys=["test_energy_calculation", "tec"]
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
        # cmd.add_command(["compute_profiles", "cp"], "Computes profiles based on a corpus.", self._profile_creation_test)
        cmd.add_command(["process_uncertain", "pu"], "Processes the uncertain keys interactively.", self._uncertain_keys_processor.process)
        cmd.add_command(["print_unmatched", "pru"], "Prints the library entries whose UUIDs don't have an associated cached scrobble.", self._unmatch_util.print_unmatched_tracks)
        cmd.add_command(["build_cache", "bc"], "Builds the cache for the library for n samples.", self._build_cache, arguments=[False])
        cmd.add_command(["build_cache-profile", "bc-p"], "Builds the cache for the library for n samples and profiles the performance.", self._build_cache, arguments=[True])
        cmd.add_command(["link_scrobbles", "ls"], "Links the scrobbles to the library data.", self._link_scrobbles, arguments=[False])
        cmd.add_command(["link_scrobbles-profile", "ls-p"], "Links the scrobbles to the library data but profiles the performance also.", self._link_scrobbles, arguments=[True])
        cmd.add_command(["download_and_md", "damd"], "Combines downloading and setting metadata.", self._download_and_assign_metadata)
        cmd.add_command(["add_album_to_downloads", "aatd"], "Adds an album to the downloads.csv file.", self._metadata_api.add_album_to_downloads)
        cmd.add_command(["log_hdf5_stats", "lhs"], "Logs interesting statistics for the hdf5 cache.", self._log_hdf5_stats)
        cmd.add_command(["save_directory_tree", "sdt"], "Saves the directory tree to a file.", self._save_directory_tree)

        # noinspection PyBroadException
        try:
            cmd.start_listen_loop()
        except Exception as _:
            tb = traceback.format_exc()
            self._logger.critical(
                f"Something went terribly wrong, causing the application to nearly crash. Restarting.\n{tb}"
            )
            self.run()

    @staticmethod
    def _save_directory_tree():
        config: DirectoryTreeConfig = DirectoryTreeConfig()
        generator: DirectoryTreeGenerator = DirectoryTreeGenerator(config)

        root_path: Path = Path(r"X:\Track Analysis\track_analysis\components\track_analysis")
        output_path = root_path / "directory_tree.txt"
        generator.generate(root_path, output_path)

    @staticmethod
    def _log_hdf5_stats():
        MEMORY.log_stats(top_n=5)

    # def _profile_creation_test(self):
    #     corpus_path = Path(r"X:\Track Analysis\data\training\corpus.csv")
    #
    #     def __run():
    #         self._profile_creator.generate_profile(corpus_path)
    #
    #     __run()

    def _download_and_assign_metadata(self):
        # Run the download pipeline
        data: DownloadPipelineContext = self._download_pipeline.flow(
            DownloadPipelineContext(download_csv_path=self._configuration.paths.download_csv_file)
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

    def on_exit(self) -> None:
        self._exit()

    def _exit(self):
        MEMORY.close()
        self._max_rate_cache.save()
        self._logger.save()
        exit()

    def _generate_embeddings(self):
        self._scrobble_linker.build_embeddings_for_library()

    def _build_cache(self, profiling: bool = False) -> None:
        if not profiling:
            self._scrobble_linker.build_cache()
            return

        _run_with_profiling(self._scrobble_linker.build_cache, "Embedding Construction", self._configuration.paths.benchmark_directory)

    def _link_scrobbles(self, profiling: bool = False) -> None:
        output_path: Path = self._configuration.paths.enriched_scrobble_data

        if self._configuration.development.delete_final_data_before_start:
            output_path.unlink(missing_ok=True)

        def _perform_linking() -> Any:
            enriched = self._scrobble_linker.link_scrobbles()
            enriched.to_csv(output_path, index=False)
            return enriched

        if profiling:
            _run_with_profiling(_perform_linking, "Scrobble Matching", self._configuration.paths.benchmark_directory)
            return

        # Regular execution
        _perform_linking()

    def _make_csv(self, profile: bool = False) -> None:
        key_progression_path: Path = self._configuration.paths.key_progression_data

        pipeline_context = LibraryDataGenerationPipelineContext(
            source_dir=self._configuration.paths.root_music_library,
            main_data_output_file_path=self._library_data_path,
            key_progression_output_file_path=key_progression_path,
            mfcc_data_output_file_path=self._mfcc_data_path,
            use_threads=True,
            max_new_tracks_per_run=self._configuration.additional_config.max_new_tracks_per_run,
            missing_headers_to_fill=[
                Header.Onset_Rate_Variation,
                Header.High_Ratio,
                Header.Mid_Ratio,
                Header.Low_Mid_Ratio,
                Header.Bass_Ratio,
                Header.Sub_Bass_Ratio,
                Header.Spectral_Bandwidth_STD,
                Header.Spectral_Bandwidth_Mean,
                Header.Zero_Crossing_Rate_STD,
                Header.Spectral_Entropy,
                Header.Spectral_Kurtosis,
                Header.Spectral_Skewness,
                Header.Spectral_Contrast_STD,
                Header.Spectral_Flatness_STD,
                Header.Spectral_Flux_Std,
                Header.Spectral_Centroid_Std,
                Header.Chroma_Entropy,
                Header.HPR,
                Header.Rhythmic_Regularity,
                Header.Beat_Strength,
                Header.Integrated_LUFS_Range,
                Header.Integrated_LUFS_STD,
                Header.Integrated_LUFS_Mean
            ],
            headers_to_refill=[
                Header.MFCC,
                # Header.Energy_Level
            ],
            end_at_energy_calculation_loading=False
        )

        # output_path.unlink(missing_ok=True)

        pipeline_config: PipelineConfiguration = PipelineConfiguration(num_workers=self._configuration.additional_config.num_workers_cpu_heavy-10, hop_length=512, n_fft=2048)

        pipeline = BuildLibraryDataCSVPipeline(
            logger=self._logger,
            filehandler=self._file_handler, string_utils=self._string_utils,
            tag_extractor=self._tag_extractor, max_rate_cache=self._max_rate_cache,
            configuration=pipeline_config, app_configuration=self._configuration
        )

        def __run():
            pipeline.build_pipeline()
            pipeline.flow(pipeline_context)

        if profile:
            _run_with_profiling(__run, "Building Library CSV", self._configuration.paths.benchmark_directory)
        else: __run()

        self._logger.info("CSV has been successfully created.")
