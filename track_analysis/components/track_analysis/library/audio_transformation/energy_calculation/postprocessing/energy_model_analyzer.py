import dataclasses
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model import \
    EnergyModel


@dataclasses.dataclass(frozen=True)
class AnalysisConfig:

    dominant_loading_threshold: float = 0.7
    """Above what value a feature dominates the component."""

    top_n_components: int = 3
    """The number of top n components to show."""

    top_n_features: int = 3
    """The amount of positive and negative features to show per component."""

class EnergyModelAnalyzer:
    """
    Analyzes a trained EnergyModel to provide interpretations and flag potential issues.
    """

    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = self.__class__.__name__

    def analyze(self, model: EnergyModel, analysis_config: AnalysisConfig = AnalysisConfig()) -> Dict[str, Any]:
        """
        Performs a full analysis of the model, returning interpretations and red flags.

        Returns:
            A dictionary containing the analysis report.
        """
        self._logger.info("--- Starting PCA Model Analysis ---", separator=self._separator)
        interpretations = self._interpret_components(model, analysis_config)
        red_flags = self._identify_red_flags(model, analysis_config)

        report = {
            "component_interpretations": interpretations,
            "red_flags": red_flags
        }

        self._logger.info("--- Analysis Complete ---", separator=self._separator)
        self._log_report(report)

        return report

    @staticmethod
    def _interpret_components(model: EnergyModel, analysis_config: AnalysisConfig) -> Dict[str, str]:
        interpretations = {}
        for i in range(analysis_config.top_n_components):
            component_name = f"PC{i+1}"
            variance = model.pca.explained_variance_ratio_[i] * 100

            loadings = pd.Series(model.pca.components_[i], index=model.feature_names)

            # Get top N features for the positive and negative poles of the axis
            top_positive = loadings.nlargest(analysis_config.top_n_features)
            top_negative = loadings.nsmallest(analysis_config.top_n_features)

            # Build the interpretation string
            positive_pole_desc = ", ".join([f"{feat} ({val:.2f})" for feat, val in top_positive.items()])
            negative_pole_desc = ", ".join([f"{feat} ({val:.2f})" for feat, val in top_negative.items()])

            interpretation = (
                f"Explains {variance:.2f}% of variance.\n"
                f"  - A HIGH score is associated with: [{positive_pole_desc}].\n"
                f"  - A LOW score is associated with: High values in [{negative_pole_desc}]."
            )
            interpretations[component_name] = interpretation
        return interpretations

    @staticmethod
    def _identify_red_flags(model: EnergyModel, analysis_config: AnalysisConfig) -> List[str]:
        flags = []
        for i in range(model.number_of_pca_components):
            component_name = f"PC{i+1}"
            loadings = model.pca.components_[i]

            # RED FLAG 1: Dominant Feature Loading
            max_abs_loading = np.max(np.abs(loadings))
            if max_abs_loading >= analysis_config.dominant_loading_threshold:
                dominant_feature_idx = np.argmax(np.abs(loadings))
                dominant_feature = model.feature_names[dominant_feature_idx]
                flag_desc = (
                    f"[{component_name}] has a DOMINANT FEATURE: '{dominant_feature}' "
                    f"has a loading of {loadings[dominant_feature_idx]:.3f}. "
                    f"This component is almost entirely defined by one feature, which can skew the model. "
                    f"Consider removing this feature if its meaning is ambiguous or misleading."
                )
                flags.append(flag_desc)

        return flags

    def _log_report(self, report: Dict[str, Any]) -> None:
        """Logs the analysis report to the console."""
        self._logger.info("Component Interpretations:", separator=self._separator)
        for name, desc in report['component_interpretations'].items():
            self._logger.info(f"{name}: {desc}", separator=self._separator)

        self._logger.info("----------------------------------", separator=self._separator)
        self._logger.info("Model Red Flags:", separator=self._separator)
        if not report['red_flags']:
            self._logger.info("  - No red flags identified. Model looks balanced.", separator=self._separator)
        else:
            for flag in report['red_flags']:
                self._logger.warning(f"  - WARNING: {flag}", separator=self._separator)
