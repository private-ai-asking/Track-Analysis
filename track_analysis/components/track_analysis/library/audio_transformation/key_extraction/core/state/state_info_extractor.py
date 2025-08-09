from typing import Dict, List, Optional

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.core.exceptions.lookup_exceptions import \
    MalformedLabelError, UnknownTonicError
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.core.lookup.tonic_lookup import \
    TonicLookup
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.core.state.state_info import \
    StateLabelInfo
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.core.parsing.parse_label import \
    parse_label


class StateInfoExtractor:
    def __init__(self, logger: HoornLogger, tonic_index_map: Dict[str, int]):
        self._tonic_lookup = TonicLookup(tonic_index_map)
        self._logger = logger
        self._separator = self.__class__.__name__

    def extract(self, state_labels: List[str]) -> List[StateLabelInfo]:
        """
        Iterate over raw state_labels; for each:
          1. parse into (tonic, mode)
          2. look up tonic index
          3. if both succeed, return a frozen StateLabelInfo
          4. if parsing or lookup fails, log an error and skip
        """
        valid_states: List[StateLabelInfo] = []

        for i, label in enumerate(state_labels):
            state_info = self._extract(i, label)
            if state_info: valid_states.append(state_info)

        return valid_states

    def _extract(self, idx: int, label: str) -> Optional[StateLabelInfo]:
        try:
            tonic, mode = parse_label(label)
            self._tonic_lookup.validate(tonic, label)
            return StateLabelInfo(index=idx, tonic=tonic, mode=mode)
        except MalformedLabelError as e:
            self._logger.error(str(e), separator=self._separator)
        except UnknownTonicError as e:
            self._logger.error(str(e), separator=self._separator)
