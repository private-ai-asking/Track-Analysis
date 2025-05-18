import pydantic


class CandidateModel(pydantic.BaseModel):
    lib_idx: int
    uuid: str
    distance: float
    combined_token_similarity: float
    title_token_similarity: float
    artist_token_similarity: float
    album_token_similarity: float
