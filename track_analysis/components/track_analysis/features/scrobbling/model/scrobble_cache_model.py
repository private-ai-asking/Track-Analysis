import pydantic


class ScrobbleCacheItemModel(pydantic.BaseModel):
    """Represents an item of scrobble cache in json."""
    associated_uuid: str
    associated_track_title: str
    associated_track_album: str
    associated_track_artist: str
    confidence_factor_percentage: float
