from typing import List

import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.core.definitions.definition_order_of_fifths import \
    ORDER_OF_FIFTHS
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.core.state.state_info import \
    StateLabelInfo
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.penalties.mode_penalty_calculator import \
    ModePenaltyCalculator
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.core.state.state_info_extractor import \
    StateInfoExtractor
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.penalties.tonic_penalty_calculator import \
    TonicPenaltyCalculator


class PenaltyMatrixBuilder:
    """
    Builds a penalty matrix given base penalties and states.
    """
    def __init__(
            self,
            logger: HoornLogger,
            state_labels: List[str],
            base_tonic_penalty: float,
            base_mode_penalty: float
    ):
        self._logger = logger
        self._separator = self.__class__.__name__

        # Scales penalties to fit into the -1, 1 space for pearson correlation.
        # Why divided by 38.5? Because the conceptual penalty was originally meant for that scale.
        scaled_tone_penalty = base_tonic_penalty * 2 / 38.5
        scaled_mode_penalty = base_mode_penalty * 2 / 38.5

        extractor = StateInfoExtractor(logger, {t: i for i, t in enumerate(ORDER_OF_FIFTHS)})
        self._valid_states: List[StateLabelInfo] = extractor.extract(state_labels)

        self._tonic_distance_calculator = TonicPenaltyCalculator(scaled_tone_penalty)
        self._mode_penalty_calculator = ModePenaltyCalculator(
            logger, scaled_mode_penalty
        )

        self._n = len(state_labels)
        self._logger.debug(
            f"Initialized with {self._n} states, tonic_penalty={scaled_tone_penalty},",
            separator=self._separator,
        )

    def build(self) -> np.ndarray:
        """
        Build an advanced penalty matrix of size (n × n).
        """
        self._logger.info("Building advanced penalty matrix...", separator=self._separator)

        matrix = np.zeros((self._n, self._n), dtype=float)

        for state in self._valid_states:
            for other_state in self._valid_states:
                self._compute_and_log_cell(matrix, state.index, other_state.index, state.tonic, state.mode, other_state.tonic, other_state.mode)

        self._logger.info("Completed advanced penalty matrix.", separator=self._separator)
        return matrix

    def _compute_and_log_cell(
            self,
            matrix: np.ndarray,
            i: int,
            j: int,
            tonic_i: str,
            mode_i: str,
            tonic_j: str,
            mode_j: str
    ) -> None:
        """
        Compute penalty for (i, j), write into matrix[i, j], and emit a trace log.
        """
        tonic_penalty = self._tonic_distance_calculator.penalty_between(tonic_i, tonic_j)
        mode_penalty = self._mode_penalty_calculator.penalty_between(mode_i, mode_j)
        total_penalty = tonic_penalty + mode_penalty

        matrix[i, j] = total_penalty

        self._logger.trace(
            f"Penalty[{i},{j}] = "
            f"tonic({tonic_i}↔{tonic_j}: {tonic_penalty:.2f}) + "
            f"mode({mode_i}->{mode_j}: {mode_penalty:.2f}) = {total_penalty:.2f}",
            separator=self._separator,
        )
