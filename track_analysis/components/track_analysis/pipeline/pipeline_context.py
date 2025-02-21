from pathlib import Path
from typing import List, Optional

import pydantic

from track_analysis.components.track_analysis.model.album_cost import AlbumCostModel
from track_analysis.components.track_analysis.model.audio_info import AudioInfo


class PipelineContextModel(pydantic.BaseModel):
    source_dir: Path
    output_file_path: Path
    album_costs: List[AlbumCostModel]
    audio_file_paths: Optional[List[Path]] = []
    audio_info: Optional[List[AudioInfo]] = []
