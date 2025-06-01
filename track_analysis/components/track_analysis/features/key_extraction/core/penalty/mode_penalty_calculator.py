from typing import Dict

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.key_extraction.core.definitions.definition_penalties import \
    MODE_PENALTIES


class ModePenaltyCalculator:
    """
    Encapsulates lookup and scaling of mode‐transition penalties.
    """
    def __init__(self, logger: HoornLogger, base_mode_penalty: float):
        self._logger = logger
        self._separator = self.__class__.__name__

        self._base_mode_penalty = base_mode_penalty
        self._mode_matrix: Dict[str, Dict[str, float]] = MODE_PENALTIES

    def penalty_between(self, mode_i: str, mode_j: str) -> float:
        raw = self._mode_matrix.get(mode_i, {}).get(mode_j)
        if raw is None:
            self._logger.warning(
                f"No mode penalty for {mode_i} -> {mode_j}; default 0",
                separator=self._separator
            )
            return 0.0
        return raw * self._base_mode_penalty
