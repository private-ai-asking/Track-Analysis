from typing import Dict, List, Optional, Any

import faiss
import numpy as np
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

    library_keys: List[str]

    library_index: faiss.Index

    auto_accepted_scrobbles: Optional[pd.DataFrame] = None
    auto_rejected_scrobbles: Optional[pd.DataFrame] = None
    confused_scrobbles: Optional[pd.DataFrame] = None

    gold_standard_df: Optional[pd.DataFrame] = None

    dynamic_accept_threshold: Optional[float] = None
    dynamic_reject_threshold: Optional[float] = None

    library_row_lookup: Optional[Dict] = None

    model_config = {
        "arbitrary_types_allowed": True
    }

    def model_post_init(self, __context: Any) -> None:
        # 1) build your fast UUID → row cache
        self.library_row_lookup = (
            self.library_data_frame[["UUID", "_n_title", "_n_artist", "_n_album"]]
            .set_index("UUID")
            .to_dict(orient="index")
        )
