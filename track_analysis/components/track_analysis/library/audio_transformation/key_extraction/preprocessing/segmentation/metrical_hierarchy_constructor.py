from typing import List, Tuple

import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import VERBOSE


class MetricalHierarchyConstructor:
    """
    Constructs a metrical hierarchy by assigning levels to sub-beat events.
    Each event is given a level from 1 to LEVEL_COUNT based on divisors of events per bar.
    """
    LEVEL_COUNT = 4

    def __init__(
            self,
            subdivisions_per_beat: int,
            logger: HoornLogger,
            separator: str,
    ) -> None:
        self._subdivisions = subdivisions_per_beat
        self._logger = logger
        self._separator = separator

    def construct_hierarchy(
            self,
            beats_per_segment: int,
            sub_beat_events: List[Tuple[float, int]]
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Generate a level array and corresponding event times for sub-beat events.
        """
        self._validate_inputs(beats_per_segment)

        events_per_bar = self._calculate_events_per_segment(beats_per_segment)
        top_divisors = self._get_top_divisors(events_per_bar)

        levels, event_times = self._assign_event_levels(
            sub_beat_events, events_per_bar, top_divisors
        )

        self._log_levels(levels)
        return levels, event_times

    def _validate_inputs(
            self,
            beats_per_segment: int,
    ) -> None:
        if self._subdivisions < 1:
            raise ValueError("subdivisions_per_beat must be positive")
        if beats_per_segment < 1:
            raise ValueError("beats_per_segment must be positive")

    def _calculate_events_per_segment(self, beats_per_segment: int) -> int:
        return beats_per_segment * self._subdivisions

    def _get_top_divisors(self, events_per_segment: int) -> List[int]:
        divisors = self._get_divisors(events_per_segment)
        if len(divisors) < self.LEVEL_COUNT:
            raise ValueError(
                f"Insufficient divisors ({len(divisors)}) for {self.LEVEL_COUNT} levels; "
                "increase subdivisions_per_beat or adjust beats_per_bar"
            )
        return divisors[: self.LEVEL_COUNT]

    @staticmethod
    def _get_divisors(n: int) -> List[int]:
        """Return all positive divisors of n, sorted descending."""
        return sorted([d for d in range(1, n + 1) if n % d == 0], reverse=True)

    def _assign_event_levels(
            self,
            events: List[Tuple[float, int]],
            events_per_segment: int,
            divisors: List[int],
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Assign levels to each event based on its position modulo events_per_bar.
        Higher levels correspond to coarser metrical positions.
        """
        levels: List[int] = []
        times: List[float] = []

        for idx, (time_point, _) in enumerate(events):
            position = idx % events_per_segment
            level = self._determine_level(position, divisors)
            levels.append(level)
            times.append(time_point)

        return np.array(levels, dtype=int), times

    @staticmethod
    def _determine_level(position: int, divisors: List[int]) -> int:
        """
        Determine event level by finding the first divisor that evenly divides position.
        Level is higher for larger divisors (coarser beats).
        """
        for idx, div in enumerate(divisors):
            if position % div == 0:
                return MetricalHierarchyConstructor.LEVEL_COUNT - idx
        return 1

    def _log_levels(self, levels: np.ndarray) -> None:
        if VERBOSE:
            self._logger.debug(
                f"Assigned levels: {levels.tolist()}", separator=self._separator
            )
