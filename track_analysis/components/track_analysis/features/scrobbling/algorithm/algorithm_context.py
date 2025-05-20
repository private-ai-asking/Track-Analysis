from typing import Dict, List, Optional, Any

import faiss
import numpy as np
import pandas as pd
import pydantic


class CacheBuildingAlgorithmContext(pydantic.BaseModel):
    """Context for the Cache Building Algorithm."""
    original_scrobble_count: int
    previous_pipe_description: str

    scrobble_data_frame: pd.DataFrame
    library_data_frame: pd.DataFrame
    library_lookup_key_to_uuid: Dict[str, str]
    uncertain_keys: List[str]

    library_keys: List[str]

    library_index: faiss.Index

    manual_override_lookup: Optional[Dict[str, str]] = None

    auto_accepted_scrobbles: Optional[pd.DataFrame] = None
    auto_rejected_scrobbles: Optional[pd.DataFrame] = None
    confused_scrobbles: Optional[pd.DataFrame] = None

    gold_standard_df: Optional[pd.DataFrame] = None

    dynamic_accept_threshold: Optional[float] = None
    dynamic_reject_threshold: Optional[float] = None

    model_config = {
        "arbitrary_types_allowed": True
    }
