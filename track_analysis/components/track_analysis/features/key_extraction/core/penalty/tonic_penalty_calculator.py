from typing import List

from track_analysis.components.track_analysis.features.key_extraction.core.definitions.definition_order_of_fifths import \
    ORDER_OF_FIFTHS


class TonicPenaltyCalculator:
    """
    Given a circle (list of tonics in order) and a base penalty weight,
    computes the distance penalty between any two tonic indices.
    """
    def __init__(self, base_tonic_penalty: float):
        self._circle: List[str] = ORDER_OF_FIFTHS
        self._base_tonic_penalty = base_tonic_penalty
        self._tonic_index = {t: i for i, t in enumerate(ORDER_OF_FIFTHS)}

    def penalty_between(self, tonic_i: str, tonic_j: str) -> float:
        idx_i = self._tonic_index[tonic_i]
        idx_j = self._tonic_index[tonic_j]
        diff = abs(idx_i - idx_j)
        size = len(self._circle)
        tonic_dist = min(diff, size - diff)
        return self._base_tonic_penalty * tonic_dist
