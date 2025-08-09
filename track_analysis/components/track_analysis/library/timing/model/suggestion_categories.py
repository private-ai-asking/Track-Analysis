from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class SuggestionCategories:
    """A container for all generated suggestion types."""
    wait_candidates: List[Tuple[str, str]] = field(default_factory=list)
    optimize_candidates: List[Tuple[str, str]] = field(default_factory=list)
    variance_candidates: List[Tuple[str, str]] = field(default_factory=list)
    caching_candidates: List[Tuple[str, str]] = field(default_factory=list)
