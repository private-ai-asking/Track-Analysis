from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.helpers.cache_updater import CacheUpdater
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext


class FillMissingHeadersPipe(IPipe):
    """
    Fills rows with missing data for specified headers using a delegated processing strategy.
    This pipe identifies headers with missing values, recalculates them, and merges the
    updated data back into the main cache.
    """

    _SEPARATOR = "BuildCSV.FillMissingHeaders"

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
        """
        Executes the logic to fill missing header data.

        Args:
            context (LibraryDataGenerationPipelineContext): The pipeline context.

        Returns:
            LibraryDataGenerationPipelineContext: The updated pipeline context.
        """
        if not context.missing_headers:
            self._logger.debug("No headers with missing data to fill.", separator=self._SEPARATOR)
            return context

        headers_to_fill = set(context.missing_headers.keys())
        self._logger.info(f"Found {len(headers_to_fill)} headers with missing data.", separator=self._SEPARATOR)

        if not headers_to_fill:
            self._logger.info("All missing headers were handled by specialized processors.", separator=self._SEPARATOR)
            return context

        self._logger.info(f"Refilling {len(headers_to_fill)} standard headers via main processor.", separator=self._SEPARATOR)

        uuids_to_process = {uuid for header in headers_to_fill for uuid in context.missing_headers.get(header, [])}
        if not uuids_to_process:
            self._logger.warning("Standard headers need filling, but no associated UUIDs found.", separator=self._SEPARATOR)
            return context

        self._cache_updater.update_cache(context, list(uuids_to_process), list(headers_to_fill))

        self._logger.info("Successfully filled standard headers and updated the cache.", separator=self._SEPARATOR)
        return context
