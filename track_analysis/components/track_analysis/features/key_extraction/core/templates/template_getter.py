from typing import Dict

import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.key_extraction.core.definitions.definition_templates import \
    TemplateMode, TEMPLATE_REGISTRY


class TemplateGetter:
    """
    Retrieves a fixed mapping of scale → NumPy‐array for a given Mode.
    """
    def __init__(self, logger: HoornLogger, mode: TemplateMode):
        self._logger = logger
        self._separator = self.__class__.__name__

        if mode not in TEMPLATE_REGISTRY:
            msg = f"No templates registered for mode '{mode.value}'."
            self._logger.error(msg, separator=self._separator)
            raise ValueError(msg)

        self._mode = mode

    def get_templates(self) -> Dict[str, np.ndarray]:
        """
        Return the scale→array mapping for the configured mode.
        """
        return TEMPLATE_REGISTRY[self._mode]
