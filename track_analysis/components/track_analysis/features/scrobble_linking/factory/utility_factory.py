from typing import Dict

from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.scrobble_linking.utils.scrobble_utility import ScrobbleUtility


class ScrobbleUtilityFactory:
    """Creates a scrobble utility service."""
    def __init__(self, logger: HoornLogger, embedder: SentenceTransformer):
        self._logger = logger
        self._embedder = embedder

    def create(self, embed_weights: Dict[str, float], embed_batch_size: int, join_key: str) -> ScrobbleUtility:
        return ScrobbleUtility(
            logger=self._logger,
            embedder=self._embedder,
            embed_weights=embed_weights,
            embed_batch_size=embed_batch_size,
            join_key=join_key,
        )
