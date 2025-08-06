from typing import Dict

from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.core.exceptions.lookup_exceptions import \
    UnknownTonicError


class TonicLookup:
    """
    Encapsulates the tonic-index mapping.
    If tonic not found, raises UnknownTonicError.
    """
    def __init__(self, tonic_index_map: Dict[str, int]):
        self._tonic_index_map = tonic_index_map

    def find(self, tonic: str, original_label: str) -> int:
        idx = self._tonic_index_map.get(tonic)
        if idx is None:
            raise UnknownTonicError(tonic, original_label)
        return idx

    def validate(self, tonic: str, original_label: str) -> None:
        """Validates whether a tonic/original label are valid. Throws UnknownTonicError if not."""
        self.find(tonic, original_label)
