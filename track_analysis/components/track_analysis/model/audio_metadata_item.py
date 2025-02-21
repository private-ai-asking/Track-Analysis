import typing

import pydantic

from track_analysis.components.track_analysis.model.header import Header


class AudioMetadataItem(pydantic.BaseModel):
    """
    AudioInfoModel represents the structure of audio metadata.
    """
    header: Header
    description: str
    value: typing.Any
