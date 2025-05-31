import numpy as np
from typing import List, Dict, Tuple

from track_analysis.components.md_common_python.py_common.logging import HoornLogger


class PenaltyMatrixBuilder:
    """
    Builds a transition penalty matrix that accounts for both:
      - Tonic distance around a circle (e.g., circle of fifths)
      - Mode change costs based on empirical or provided mode-transition weights

    Total penalty[i,j] = w_t * tonic_distance(i->j) + mode_penalty[mode_i][mode_j]
    """
    def __init__(
            self,
            logger: HoornLogger,
            state_labels: List[str],
            circle_of_fifths: List[str],
            base_tonic_penalty: float,
            base_mode_penalty: float,
            mode_penalty_matrix: Dict[str, Dict[str, float]]
    ):
        """
        :param logger: logger instance
        :param state_labels: e.g. "C Ionian (Major)"
        :param circle_of_fifths: list of tonics in circle order
        :param base_tonic_penalty: multiplier per step on circle
        :param mode_penalty_matrix: nested dict mapping mode_i->mode_j->penalty
        """
        self._logger = logger
        self._separator = self.__class__.__name__
        self._state_labels = state_labels
        self._circle = circle_of_fifths
        self._base_tonic_penalty = base_tonic_penalty
        self._base_mode_penalty = base_mode_penalty
        self._mode_penalty = mode_penalty_matrix

        # index maps
        self._tonic_index: Dict[str,int] = {t: i for i, t in enumerate(circle_of_fifths)}
        self._n = len(state_labels)

        self._logger.debug(
            f"Initialized with {self._n} states, tonic_penalty={base_tonic_penalty},",
            separator=self._separator
        )

    def build(self) -> np.ndarray:
        """
        Construct the (n_states x n_states) penalty matrix.
        """
        self._logger.info("Building advanced penalty matrix...", separator=self._separator)
        matrix = np.zeros((self._n, self._n), dtype=float)

        for i, label_i in enumerate(self._state_labels):
            tonic_i, mode_i = self._split_label(label_i)
            idx_i = self._tonic_index.get(tonic_i)
            if idx_i is None:
                self._logger.error(f"Unknown tonic '{tonic_i}' in '{label_i}'", separator=self._separator)
                continue

            for j, label_j in enumerate(self._state_labels):
                tonic_j, mode_j = self._split_label(label_j)
                idx_j = self._tonic_index.get(tonic_j)
                if idx_j is None:
                    self._logger.error(f"Unknown tonic '{tonic_j}' in '{label_j}'", separator=self._separator)
                    continue

                # tonic distance
                diff = abs(idx_i - idx_j)
                tonic_dist = min(diff, len(self._circle) - diff)
                tonic_pen = self._base_tonic_penalty * tonic_dist

                # mode penalty
                mode_pen = self._mode_penalty.get(mode_i, {}).get(mode_j)
                if mode_pen is None:
                    self._logger.warning(
                        f"No mode penalty for {mode_i}->{mode_j}; default 0", separator=self._separator
                    )
                    mode_pen = 0.0

                mode_pen = mode_pen * self._base_mode_penalty

                total_pen = tonic_pen + mode_pen
                matrix[i, j] = total_pen
                self._logger.trace(
                    f"Penalty[{i},{j}] = tonic({tonic_dist}*{self._base_tonic_penalty}) + mode({mode_pen}) = {total_pen}",
                    separator=self._separator
                )

        self._logger.info("Completed advanced penalty matrix.", separator=self._separator)
        return matrix

    def _split_label(self, label: str) -> Tuple[str, str]:
        """
        Split a label into (tonic, mode). Assumes first token is tonic.
        """
        parts = label.split(maxsplit=1)
        if len(parts) < 2:
            self._logger.error(f"Invalid state label format: '{label}'", separator=self._separator)
            return parts[0], ''
        return parts[0], parts[1]
