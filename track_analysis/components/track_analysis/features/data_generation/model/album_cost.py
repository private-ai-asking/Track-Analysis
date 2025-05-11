import pydantic


class AlbumCostModel(pydantic.BaseModel):
    Album_Title: str
    Album_Cost: float
