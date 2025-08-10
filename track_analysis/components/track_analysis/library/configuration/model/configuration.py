import dataclasses
from pathlib import Path
from typing import Dict, List

import psutil

DATA_DIRECTORY_INTERNAL_DEFAULT: Path = Path("X:/Track Analysis/data/__internal__")


@dataclasses.dataclass(frozen=True)
class PathConfig:
    library_data: Path
    mfcc_data: Path
    scrobble_data: Path
    scrobble_cache: Path
    enriched_scrobble_data: Path
    key_progression_data: Path
    gold_standard_data: Path
    manual_override: Path
    music_track_downloads: Path
    library_keys: Path
    max_rate_cache: Path
    root_music_library: Path
    benchmark_directory: Path
    ffmpeg_path: Path
    cookies_file: Path
    download_csv_file: Path
    cache_dir: Path
    expensive_cache_dir: Path


@dataclasses.dataclass(frozen=True)
class ScrobbleLinkerConfig:
    # --- Existing Fields ---
    embedding_weights: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {'title': 0.35, 'artist': 0.4, 'album': 0.25}
    )
    field_combination_key: str = "||"
    token_accept_threshold: float = 70.0
    embedding_batch_size: int = 64
    # --- New Fields ---
    embedder_path: Path = DATA_DIRECTORY_INTERNAL_DEFAULT / "all-MiniLM-l12-v2-embed"
    scrobble_index_dir: Path = DATA_DIRECTORY_INTERNAL_DEFAULT


@dataclasses.dataclass(frozen=True)
class DevelopmentConfig:
    # --- New Fields ---
    debug: bool = False
    verbose: bool = False
    profile_data_loading: bool = False
    clear_cache: bool = False
    delete_final_data_before_start: bool = False
    test_cache_builder_mode: bool = False


@dataclasses.dataclass(frozen=True)
class AdditionalConfiguration:
    test_sample_size: int = -1
    supported_music_extensions: List[str] = dataclasses.field(
        default_factory=lambda: [".mp3", ".flac", ".m4a"]
    )
    energy_calculation_regenerate_library_growth_perc: float = 0.1
    current_energy_training_version_to_use: int = -1
    number_of_mfccs: int = 20
    no_match_label: str = "<NO MATCH>"
    num_workers_cpu_heavy: int = psutil.cpu_count(logical=True) + 10
    max_new_tracks_per_run: int = 50
    keys_to_be_ignored_in_cache_check: List[str] = dataclasses.field(
        default_factory=list
    )


@dataclasses.dataclass(frozen=True)
class TrackAnalysisConfigurationModel:
    paths: PathConfig
    scrobble_linker: ScrobbleLinkerConfig
    development: DevelopmentConfig
    additional_config: AdditionalConfiguration
