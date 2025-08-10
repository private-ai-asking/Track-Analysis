from pathlib import Path

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger


class CacheBuilderFactory:
    """Creates the cache builder."""
    def __init__(self, logger: HoornLogger, cache_path: Path):
        self._logger = logger
        self._cache_path = cache_path

    def create(self, tree_separator: str = ".") -> CacheBuilder:
        return CacheBuilder(
            logger=self._logger,
            cache_json_path=self._cache_path,
            tree_separator=tree_separator,
        )
