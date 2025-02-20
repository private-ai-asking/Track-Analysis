import typing

import pydantic


class AudioInfoModel(pydantic.BaseModel):
    """
    AudioInfoModel represents the structure of audio metadata.
    """
    header: str
    description: str
    value: typing.Any
