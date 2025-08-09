from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model import \
    EnergyModel


class DefaultEnergyModelValidator:
    """Encapsulates the logic for validating a loaded EnergyModel."""

    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = self.__class__.__name__

    def is_valid(self, model: EnergyModel | None, current_data_hash: str) -> bool:
        """
        Checks if a model is valid for use.

        A model is considered valid if it exists and its data hash matches the
        current data hash (or if validation is skipped).

        Args:
            model: The loaded EnergyModel object, or None if it doesn't exist.
            current_data_hash: The hash of the data we want to use the model with.

        Returns:
            True if the model is valid, False otherwise.
        """
        if model is None:
            self._logger.info("Validation failed: Model does not exist in cache.", separator=self._separator)
            return False

        if model.data_hash != current_data_hash:
            self._logger.warning(f"Validation failed: Data hash mismatch.", separator=self._separator)
            return False

        self._logger.info("Model validation successful.", separator=self._separator)
        return True
