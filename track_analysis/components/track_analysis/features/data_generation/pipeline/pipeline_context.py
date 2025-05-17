from pathlib import Path
from typing import List, Optional

import pydantic

from track_analysis.components.track_analysis.features.data_generation.model.album_cost import AlbumCostModel
from track_analysis.components.track_analysis.features.data_generation.model.audio_info import AudioInfo


class PipelineContextModel(pydantic.BaseModel):
    source_dir: Path
    main_data_output_file_path: Path

    loaded_audio_info_cache: Optional[List[AudioInfo]] = []

    # Generated Along the Line
    album_costs: Optional[List[AlbumCostModel]] = []
    all_audio_file_paths: Optional[List[Path]] = []
    filtered_audio_file_paths: Optional[List[Path]] = []
    invalid_cached_paths: Optional[List[Path]] = []
    generated_audio_info: Optional[List[AudioInfo]] = []
