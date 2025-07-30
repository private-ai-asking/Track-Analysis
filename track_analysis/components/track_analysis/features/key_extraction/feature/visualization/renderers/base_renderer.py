from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from matplotlib import pyplot as plt


class Renderer(ABC):
    """Base interface for rendering data onto a Matplotlib Axes."""
    @abstractmethod
    def render(self, ax: plt.Axes, data: np.ndarray) -> Any:
        ...
