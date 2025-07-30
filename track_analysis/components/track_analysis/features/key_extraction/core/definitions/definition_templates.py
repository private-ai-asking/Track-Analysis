from enum import Enum
from typing import Dict

import numpy as np


class TemplateMode(Enum):
    KS_T_REVISED = "K&S-T Revised"
    BELLMAN_BUDGE = "Bellman/Budge"
    HOORN = "Hoorn"

_KS_T_REVISED_TEMPLATES: Dict[str, np.ndarray] = {
    'Ionian (Major)': np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]),
    'Aeolian (Minor)': np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]),
}

_BELLMAN_BUDGE_TEMPLATES: Dict[str, np.ndarray] = {
    'Ionian (Major)': np.array([16.80, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 20.28, 1.80, 8.04, 0.62, 10.57]),
    'Aeolian (Minor)': np.array([18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 21.07, 7.49, 1.53, 0.92, 10.21]),
}

# 01 June 2025 V1
# _HOORN_TEMPLATES: Dict[str, np.ndarray] = {
#     'Ionian (Major)': np.array([18.38, 1.58, 18.50, 3.21, 8.35, 5.25, 2.23, 20.65, 1.33, 12.41, 2.84, 5.27]),
#     'Aeolian (Minor)': np.array([22.34, 1.59, 10.25, 9.86, 1.54, 16.47, 1.43, 21.62, 4.91, 1.81, 6.49, 1.68]),
# }

# 01 June 2025 V2
# _HOORN_TEMPLATES: Dict[str, np.ndarray] = {
#     'Ionian (Major)': np.array([14.72, 0.92, 17.63, 1.13, 12.76, 6.98, 2.59, 18.59, 0.59, 15.60, 1.32, 7.17]),
#     'Aeolian (Minor)': np.array([22.23, 1.15, 9.58, 13.22, 1.28, 15.39, 1.07, 21.29, 6.16, 1.60, 5.45, 1.57]),
#     'Dorian (Minor)': np.array([36.19, 0.87, 8.49, 8.55, 1.54, 7.78, 1.03, 19.54, 0.94, 4.93, 8.86, 1.27])
# }

# 02 June 2025 V3
# _HOORN_TEMPLATES: Dict[str, np.ndarray] = {
#     'Ionian (Major)': np.array([14.90, 3.03, 15.99, 1.67, 13.58, 8.80, 2.80, 16.92, 1.63, 10.65, 1.98, 8.04]),
#     'Aeolian (Minor)': np.array([21.73, 1.58, 9.71, 10.68, 1.82, 17.02, 1.96, 18.23, 5.96, 2.69, 6.68, 1.93]),
#     'Dorian (Minor)': np.array([31.46, 2.13, 8.40, 7.40, 2.87, 8.48, 1.89, 17.17, 3.79, 5.50, 8.79, 2.12])
# }

# 02 June 2025 V4
_HOORN_TEMPLATES: Dict[str, np.ndarray] = {
    'Ionian (Major)': np.array([15.51, 2.76, 14.09, 2.13, 12.94, 9.17, 3.21, 16.31, 1.87, 11.66, 2.27, 8.08]),
    'Aeolian (Minor)': np.array([21.65, 1.33, 10.06, 11.35, 1.95, 14.62, 1.59, 19.54, 5.96, 2.84, 6.97, 2.15]),
    # 'Dorian (Minor)': np.array([31.46, 2.13, 8.40, 7.40, 2.87, 8.48, 1.89, 17.17, 3.79, 5.50, 8.79, 2.12])
}

TEMPLATE_REGISTRY: Dict[TemplateMode, Dict[str, np.ndarray]] = {
    TemplateMode.KS_T_REVISED: _KS_T_REVISED_TEMPLATES,
    TemplateMode.BELLMAN_BUDGE: _BELLMAN_BUDGE_TEMPLATES,
    TemplateMode.HOORN: _HOORN_TEMPLATES,
}
