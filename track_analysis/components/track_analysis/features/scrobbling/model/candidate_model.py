from enum import Enum
from typing import Optional

import pydantic

class DecisionBin(Enum):
    ACCEPT = 1
    REJECT = 2
    UNCERTAIN = 3


class CandidateModel(pydantic.BaseModel):
    lib_idx: int
    uuid: str
    distance: float
    combined_token_similarity: float
    title_token_similarity: float
    artist_token_similarity: float
    album_token_similarity: float

    associated_confidence: Optional[float] = None
    passed_demands: Optional[bool] = None

    decision_bin: Optional[DecisionBin] = None
