import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.model.energy_model import \
    EnergyModel


class EnergyModelProcessor:
    """
    A stateless utility that encapsulates the shared data transformation logic
    for the energy calculation model.

    This class provides a set of methods to process raw features through the
    various stages of a trained EnergyModel (scaling, PCA, and score weighting).
    By centralizing this logic, it ensures that the exact same transformation
    pipeline is used during both model training and prediction, preventing

    divergence between the two processes.
    """

    @staticmethod
    def calculate_energy_score(composite_score: np.ndarray, energy_model: EnergyModel) -> float:
        """
        Calculates the final 1-10 energy rating from a composite PCA score.

        This method applies the trained spline to the composite score and then
        scales the output through a sigmoid function to constrain it to the
        final 1-10 range.

        Args:
            composite_score: The weighted composite score generated from the PCA components.
            energy_model: The trained EnergyModel containing the spline.

        Returns:
            The final energy score, rounded to one decimal place.
        """
        initial_rating = energy_model.spline(composite_score)
        # Scale the spline output to a range suitable for the sigmoid function
        scaled_rating = (initial_rating - 5.5) * (10 / 9)
        sigmoid_output = expit(scaled_rating)
        # Map the sigmoid output (0 to 1) to the final energy scale (1 to 10)
        final_energy = 1 + sigmoid_output * 9

        return round(float(np.clip(final_energy, 1.0, 10.0)), 1)

    def transform_to_composite_score(self, energy_model: EnergyModel, features_df: pd.DataFrame) -> np.ndarray:
        """
        Executes the full feature transformation pipeline on a DataFrame.

        This is the primary method used during prediction. It takes raw features
        and processes them through the scaler and PCA from the trained model
        to produce the final composite score.

        Args:
            energy_model: The fully trained EnergyModel.
            features_df: A DataFrame containing the raw input features.

        Returns:
            A NumPy array representing the weighted composite PCA scores.
        """
        scaled_features = self.scale_features(energy_model.scaler, features_df)
        pca_scores = self.get_pca_scores(energy_model.pca, scaled_features)
        weighted_scores = self.compute_composite_score(
            energy_model.pca, pca_scores, energy_model.number_of_pca_components, energy_model.cumulative_variance
        )
        return weighted_scores

    def compute_composite_score(self, pca: PCA, pca_scores: np.ndarray, num_components: int, cumulative_variance: float) -> np.ndarray:
        """
        Computes a single composite score from multiple PCA component scores.

        The composite score is a weighted sum of the scores from the relevant
        principal components, where each score is weighted by its explained
        variance ratio.

        Args:
            pca: The fitted PCA object from the model.
            pca_scores: The full array of scores for all PCA components.
            num_components: The number of components to include in the calculation.
            cumulative_variance: The cumulative variance of the components.

        Returns:
            A 1D NumPy array of the final composite scores.
        """
        # Weights are the percentage of variance each component explains
        raw_weights = self._get_weights(pca, num_components, cumulative_variance)
        # Scores are from the PCA transformation
        relevant_scores = pca_scores[:, :num_components]
        # The final score is the dot product of scores and their weights
        composite_scores = self._calculate_weighted_sum(relevant_scores, raw_weights)
        return composite_scores

    @staticmethod
    def get_pca_scores(pca: PCA, scaled_features: np.ndarray) -> np.ndarray:
        """Applies a fitted PCA transformation to scaled features."""
        return pca.transform(scaled_features)

    @staticmethod
    def scale_features(scaler: RobustScaler, features_df: pd.DataFrame, train: bool = False) -> np.ndarray:
        """
        Applies a scaler to raw features.

        If 'train' is True, it fits the scaler to the data before transforming.
        Otherwise, it applies the already-fitted scaler.

        Args:
            scaler: A RobustScaler instance.
            features_df: A DataFrame of raw features.
            train: If True, call fit_transform; otherwise, call transform.

        Returns:
            A NumPy array of the scaled features.
        """
        if train:
            return scaler.fit_transform(features_df)

        return scaler.transform(features_df)

    def _get_weights(self, pca: PCA, num_components: int, cumulative_variance: float) -> np.ndarray:
        """Helper for computing the weights per component."""
        relevant_variances = pca.explained_variance_ratio_[:num_components]
        return self._make_variance_ratios_sum_to_one(relevant_variances, cumulative_variance)

    @staticmethod
    def _make_variance_ratios_sum_to_one(ratios: np.ndarray, cumulative_variance: float) -> np.ndarray:
        return ratios / cumulative_variance

    @staticmethod
    def _calculate_weighted_sum(scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Helper method to compute the dot product of scores and weights."""
        return scores.dot(weights)
