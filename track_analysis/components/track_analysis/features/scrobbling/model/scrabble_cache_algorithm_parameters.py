from pathlib import Path
from typing import Dict

import pydantic


class ScrobbleCacheAlgorithmParameters(pydantic.BaseModel):
    """Encapsulation of scrobble cache algorithm tunable parameters."""
    confidence_accept_threshold: float = 60
    confidence_reject_threshold: float = 20
    token_accept_threshold: float = 70
    gaussian_sigma: float = 0.1
    batch_size: int = 64
    top_k: int = 5

    # Other
    max_gold_standard_entries: int = 150
    gold_standard_csv_path: Path
    manual_override_path: Path

    embed_weights: Dict
