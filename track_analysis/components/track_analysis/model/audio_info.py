from pathlib import Path
from typing import List

import pydantic

from track_analysis.components.track_analysis.model.audio_metadata_item import AudioMetadataItem


class AudioInfo(pydantic.BaseModel):
    path: Path
    metadata: List[AudioMetadataItem]
