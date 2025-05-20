from pathlib import Path
from typing import List, Optional

import pandas as pd
import pydantic

from track_analysis.components.track_analysis.features.audio_file_handler import AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.data_generation.model.album_cost import AlbumCostModel


class LibraryDataGenerationPipelineContext(pydantic.BaseModel):
    source_dir: Path
    main_data_output_file_path: Path
    use_threads: bool = True
    max_new_tracks_per_run: Optional[int] = 50

    loaded_audio_info_cache: Optional[pd.DataFrame] = None

    # Generated Along the Line
    album_costs: Optional[List[AlbumCostModel]] = []

    all_audio_file_paths: Optional[List[Path]] = []
    cached_audio_paths: Optional[List[Path]] = []
    filtered_audio_file_paths: Optional[List[Path]] = []
    invalid_cached_paths: Optional[List[Path]] = []

    extracted_stream_info: Optional[List[AudioStreamsInfoModel]] = []
    generated_audio_info: Optional[pd.DataFrame] = None

    model_config = {
        'arbitrary_types_allowed': True,
    }
