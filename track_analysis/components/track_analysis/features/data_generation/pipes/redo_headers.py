from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.helpers.cache_updater import CacheUpdater
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext


class RedoHeaders(IPipe):
    """
    Orchestrator class that re-calculates specified features and merges them
    back into the cached data using a dedicated updater.
    """

    _SEPARATOR = "BuildCSV.RedoHeaders"

    def __init__(self, logger: HoornLogger, cache_updater: CacheUpdater):
        """
        Initializes the pipe.

        Args:
            logger (HoornLogger): The logger instance for logging messages.
            cache_updater (CacheUpdater): The centralized handler for updating cached DataFrames.
        """
        self._logger = logger
        self._cache_updater = cache_updater
        self._logger.trace("Successfully initialized pipe.", separator=self._SEPARATOR)

    def flow(self, context: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        if not context.headers_to_refill:
            self._logger.debug("No headers to refill.", separator=self._SEPARATOR)
            return context

        self._logger.info(f"Refilling {len(context.headers_to_refill)} headers.", separator=self._SEPARATOR)
        self._cache_updater.update_cache(context, uuids_to_process=None, headers_to_update=context.headers_to_refill)

        self._logger.info("Successfully refilled headers and updated the cache.", separator=self._SEPARATOR)
        return context
