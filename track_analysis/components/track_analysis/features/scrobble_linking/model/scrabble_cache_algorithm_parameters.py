from pathlib import Path
from typing import Dict, Any, Optional

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

    uncertain_keys_path: Optional[Path] = None

    embed_weights: Dict

    def model_post_init(self, __context: Any) -> None:
        self.uncertain_keys_path: Path = self.manual_override_path.parent / "uncertain_keys_temp.csv"

