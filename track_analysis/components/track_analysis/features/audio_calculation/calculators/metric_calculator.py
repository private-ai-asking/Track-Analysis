from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

class AudioMetricCalculator(ABC):
    """Defines the interface every metric must implement."""
    @abstractmethod
    def calculate(
            self,
            audio_path: Path,
            samples: np.ndarray,
            sr: int,
            **kwargs
    ) -> dict:
        """Return a dict of metric_name → float."""
        ...
