from typing import Dict, List, Optional

import faiss
import pandas as pd
import pydantic


class AlgorithmContext(pydantic.BaseModel):
    """Context for the Cache Building Algorithm."""
    original_scrobble_count: int
    previous_pipe_description: str

    scrobble_data_frame: pd.DataFrame
    library_data_frame: pd.DataFrame
    library_lookup_key_to_uuid: Dict[str, str]
    uncertain_keys: List[str]

    library_index: faiss.Index
    library_keys: List[str]

    auto_accepted_scrobbles: Optional[pd.DataFrame] = None
    auto_rejected_scrobbles: Optional[pd.DataFrame] = None
    confused_scrobbles: Optional[pd.DataFrame] = None

    model_config = {
        "arbitrary_types_allowed": True
    }
