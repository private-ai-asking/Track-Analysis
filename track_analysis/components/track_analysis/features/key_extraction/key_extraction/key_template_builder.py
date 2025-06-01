from typing import Dict, List

import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.key_extraction.key_extraction.lof_feature_transformer import \
    LOFFeatureTransformer


class KeyTemplateBuilder:
    """
    Builds normalized key templates (chromatic profiles) for musical modes and tonics.
    """
    def __init__(
            self,
            logger: HoornLogger,
            modes: Dict[str, np.ndarray],
            tonics: List[str]
    ):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._modes = modes
        self._tonics = tonics
        self._logger.debug(f"Initialized with modes: {list(modes.keys())} and tonics: {tonics}", separator=self._separator)

    def build_templates(self) -> Dict[str, np.ndarray]:
        """
        Generate a dict mapping key names to normalized LOF-based templates.
        """
        transformer = LOFFeatureTransformer()
        templates: Dict[str, np.ndarray] = {}
        for mode_name, profile in self._modes.items():
            self._logger.trace(f"Transforming mode: {mode_name}", separator=self._separator)
            base = transformer.transform(profile)
            for shift, tonic in enumerate(self._tonics):
                key_name = f"{tonic} {mode_name}"
                rolled = np.roll(base, shift)
                templates[key_name] = rolled
                self._logger.trace(f"Built template: {key_name}", separator=self._separator)
        self._logger.info(f"Built {len(templates)} key templates.", separator=self._separator)
        return templates
