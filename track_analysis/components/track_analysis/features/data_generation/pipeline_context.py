from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import pydantic

from track_analysis.components.track_analysis.features.data_generation.processors.key_feature_processor import \
    KeyFeatureProcessor
from track_analysis.components.track_analysis.features.data_generation.processors.main_feature_processor import \
    MainFeatureProcessor
from track_analysis.components.track_analysis.legacy.audio_file_handler import AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.data_generation.model.album_cost import AlbumCostModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.energy_calculator import \
    EnergyAlgorithm
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model_config import \
    EnergyModelConfig


class LibraryDataGenerationPipelineContext(pydantic.BaseModel):
    source_dir: Path
    main_data_output_file_path: Path
    key_progression_output_file_path: Path
    mfcc_data_output_file_path: Path

    use_threads: bool = True
    end_at_energy_calculation_loading: bool = False
    max_new_tracks_per_run: Optional[int] = 50
    missing_headers_to_fill: List[Header] = []
    headers_to_refill: List[Header] = []

    loaded_audio_info_cache: Optional[pd.DataFrame] = None
    loaded_mfcc_info_cache: Optional[pd.DataFrame] = None
    loaded_key_progression_cache: Optional[pd.DataFrame] = None

    missing_headers: Optional[Dict[Header, List[str]]] = None

    # Generated Along the Line
    main_processor: MainFeatureProcessor = None
    key_processor: KeyFeatureProcessor = None

    album_costs: Optional[List[AlbumCostModel]] = []

    all_audio_file_paths: Optional[List[Path]] = []
    cached_audio_paths: Optional[List[Path]] = []
    filtered_audio_file_paths: Optional[List[Path]] = []
    invalid_cached_paths: Optional[List[Path]] = []

    energy_calculator: EnergyAlgorithm = None

    extracted_stream_info: Optional[List[AudioStreamsInfoModel]] = []
    generated_audio_info: Optional[pd.DataFrame] = None
    generated_mfcc_audio_info: Optional[pd.DataFrame] = None
    generated_key_progression_audio_info: Optional[pd.DataFrame] = None

    model_config = {
        'arbitrary_types_allowed': True,
    }
