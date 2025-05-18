from pathlib import Path
from typing import List

import pydantic

from track_analysis.components.track_analysis.features.track_downloading.model.download_model import DownloadModel


class DownloadPipelineContext(pydantic.BaseModel):
    """Context for the downloading pipeline."""
    download_csv_path: Path

    tracks_to_download: List[DownloadModel] = []
    downloaded_tracks: List[DownloadModel] = []

