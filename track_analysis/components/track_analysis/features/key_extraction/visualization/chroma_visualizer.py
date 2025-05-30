from typing import List

import matplotlib
import matplotlib.pyplot as plt

from track_analysis.components.track_analysis.features.key_extraction.visualization.model.config import VisualizationConfig
from track_analysis.components.track_analysis.features.key_extraction.visualization.util.layout_manager import LayoutManager
from track_analysis.components.track_analysis.features.key_extraction.visualization.util.time_axis_calculator import TimeAxisCalculator
from track_analysis.components.track_analysis.features.key_extraction.visualization.util.window_manager import WindowManager
from track_analysis.components.md_common_python.py_common.logging import HoornLogger

matplotlib.use("TkAgg")


class ChromaVisualizer:
    """Orchestrates rendering of multiple datasets in a single or separate layout."""

    def __init__(self, hop_length: int, logger: HoornLogger) -> None:
        self._hop_length = hop_length
        self._logger = logger

    def visualize(
            self,
            configs: List[VisualizationConfig],
            sample_rate: int,
            combined: bool = True
    ) -> None:
        time_calc = TimeAxisCalculator(self._hop_length, sample_rate)
        if combined:
            self._render_combined(configs, time_calc)
        else:
            self._render_individual(configs, time_calc)

    def _render_combined(
            self,
            configs: List[VisualizationConfig],
            time_calc: TimeAxisCalculator
    ) -> None:
        num_plots = len(configs)
        rows, cols = LayoutManager.grid(num_plots)
        fig, axes = plt.subplots(rows, cols, constrained_layout=True)
        WindowManager.maximize(fig)

        # flatten axes for consistent iteration
        axes_list = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for ax, cfg in zip(axes_list, configs):
            self._configure_and_render(ax, cfg, time_calc)

        # remove unused axes
        for ax in axes_list[num_plots:]:
            fig.delaxes(ax)

        plt.show()
        self._log_debug("Combined visualization displayed")

    def _render_individual(
            self,
            configs: List[VisualizationConfig],
            time_calc: TimeAxisCalculator
    ) -> None:
        for cfg in configs:
            fig, ax = plt.subplots(constrained_layout=True)
            WindowManager.maximize(fig)

            self._configure_and_render(ax, cfg, time_calc)
            plt.show()
            self._log_debug(f"{cfg.title} displayed separately")

    @staticmethod
    def _configure_and_render(
            ax: plt.Axes,
            cfg: VisualizationConfig,
            time_calc: TimeAxisCalculator
    ) -> None:
        # set title and labels
        ax.set(title=cfg.title, xlabel=cfg.x_label)
        ax.set(ylabel=cfg.y_label)

        # set time axis limits for 2D data
        if cfg.data.ndim > 1:
            frames = cfg.data.shape[1]
            times = time_calc.compute(frames)
            ax.set(xlim=(times[0], times[-1]))

        # render data and optional colorbar
        img = cfg.renderer.render(ax, cfg.data)
        if img is not None:
            plt.colorbar(img, ax=ax, label=cfg.color_label)

    def _log_debug(self, message: str) -> None:
        if self._logger:
            self._logger.debug(message, separator=self.__class__.__name__)
