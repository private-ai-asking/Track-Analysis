from track_analysis.components.md_common_python.py_common.logging import HoornLogger


class AudioFormatConverter:
    """Responsible for converting FFmpeg sample format strings to bit depths."""

    def __init__(self, logger: HoornLogger):
        self._separator = "AudioFormatConverter"
        self._logger = logger

    def get_bit_depth_from_format(self, sample_format: str) -> int:
        """
        Gets the bit depth from the FFmpeg sample format string.

        Args:
            sample_format: The FFmpeg sample format string (e.g., "s16le", "flt", "dbl").

        Returns:
            The bit depth as an integer (e.g., 16, 32, 64), or 0 if the format is unknown or invalid.
        """
        format_lower = sample_format.lower()

        if format_lower.startswith("s") or format_lower.startswith("u"):
            try:
                bits = int(format_lower[1:3])
                return bits
            except ValueError:
                pass
        elif format_lower.startswith("flt") or format_lower.startswith("s32p"):
            return 32
        elif format_lower.startswith("dbl"):
            return 64
        elif format_lower.startswith("s16p"):
            return 16
        elif format_lower.startswith("u8") or format_lower.startswith("s8"):
            return 8

        self._logger.warning(f"Unsupported format for bit depth extraction: {format_lower}", separator=self._separator)
        return 0
