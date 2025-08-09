from dataclasses import dataclass

import numpy as np

from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.feature.visualization.renderers.base_renderer import \
    Renderer


@dataclass(frozen=True)
class VisualizationConfig:
    data: np.ndarray
    renderer: Renderer
    title: str
    x_label: str
    y_label: str
    color_label: str = "Intensity"
