from enum import Enum
from typing import Dict

import numpy as np


class TemplateMode(Enum):
    KS_T_REVISED = "K&S-T Revised"
    BELLMAN_BUDGE = "Bellman/Budge"

_KS_T_REVISED_TEMPLATES: Dict[str, np.ndarray] = {
    'Ionian (Major)': np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]),
    'Aeolian (Minor)': np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]),
}

_BELLMAN_BUDGE_TEMPLATES: Dict[str, np.ndarray] = {
    'Ionian (Major)': np.array([16.80, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 20.28, 1.80, 8.04, 0.62, 10.57]),
    'Aeolian (Minor)': np.array([18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 21.07, 7.49, 1.53, 0.92, 10.21]),
}

TEMPLATE_REGISTRY: Dict[TemplateMode, Dict[str, np.ndarray]] = {
    TemplateMode.KS_T_REVISED: _KS_T_REVISED_TEMPLATES,
    TemplateMode.BELLMAN_BUDGE: _BELLMAN_BUDGE_TEMPLATES,
}
