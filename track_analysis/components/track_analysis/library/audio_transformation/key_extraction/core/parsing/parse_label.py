from typing import Tuple

from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.core.exceptions.lookup_exceptions import \
    MalformedLabelError


def parse_label(label: str) -> Tuple[str, str]:
    """
    Split a label into (tonic, mode).
    If it does not contain at least two whitespace-separated parts, raise MalformedLabelError.
    """
    parts = label.split(maxsplit=1)
    if len(parts) < 2:
        raise MalformedLabelError(label)
    tonic, mode = parts[0], parts[1]
    return tonic, mode
