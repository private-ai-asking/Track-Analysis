import pydantic


class ScrobbleCacheAlgorithmParameters(pydantic.BaseModel):
    """Encapsulation of scrobble cache algorithm tunable parameters."""
    confidence_accept_threshold: float = 85
    confidence_reject_threshold: float = 30
    token_accept_threshold: float = 70
    gaussian_sigma: float = 0.35
    batch_size: int = 64
    top_k: int = 2
