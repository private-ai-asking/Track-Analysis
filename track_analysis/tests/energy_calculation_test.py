from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.testing import TestInterface
from track_analysis.components.track_analysis.features.data_generation.helpers.energy_training_config_factory import \
    TrainingConfigFactory
from track_analysis.components.track_analysis.features.data_generation.helpers.energy_training_data_builder import \
    TrainingDataBuilder
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.configuration.default import \
    DEFAULT_ENERGY_MODEL_CONFIG
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.energy_calculator import \
    EnergyCalculator
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.energy_factory import \
    Implementation, EnergyFactory
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.feature_to_header_mapping import \
    FEATURE_TO_HEADER_MAPPING

TO_CHECK = [
    r"W:\media\music\[02] organized\[01] hq\OST\Path of Exile (Original Game Soundtrack)\29 Chamber of Innocence.flac",
    r"W:\media\music\[02] organized\[01] hq\OST\Path of Exile (Original Game Soundtrack)\30 High Templar Avarius.flac",
    r"W:\media\music\[02] organized\[01] hq\OST\Path of Exile (Original Game Soundtrack)\08 Merveil.flac",
    r"W:\media\music\[02] organized\[01] hq\Reggae\Damian Marley\Stony Hill (Explicit)\03 Damian “Jr. Gong” Marley - Nail Pon Cross.flac",
    r"W:\media\music\[02] organized\[01] hq\Classical\Ludovico Einaudi\Elegy For The Arctic\01 Elegy for the Arctic.flac",
    r"W:\media\music\[02] organized\[02] lq\Ambient\Tenebrae Aeternae\01 - myNoise - Ora Pro Nobis.flac",
    r"W:\media\music\[02] organized\[02] lq\Ambient\Tenebrae Aeternae\07 - myNoise - Homini Lupus.flac",
    r"W:\media\music\[02] organized\[02] lq\Ambient\Tenebrae Aeternae\02 - myNoise - Lux Noctis.flac",
    r"W:\media\music\[02] organized\[02] lq\Ambient\Tenebrae Aeternae\03 - myNoise - Verbum Maleficum.flac",
    r"W:\media\music\[02] organized\[02] lq\Ambient\Tenebrae Aeternae\04 - myNoise - Tenebrarum Tempora.flac",
    r"W:\media\music\[02] organized\[02] lq\Ambient\Tenebrae Aeternae\05 - myNoise - Pax Animis Nostris.flac",
    r"W:\media\music\[02] organized\[02] lq\Ambient\Tenebrae Aeternae\06 - myNoise - Vagabunda Anima.flac",
    r"W:\media\music\[02] organized\[02] lq\Ambient\Tenebrae Aeternae\08 - myNoise - Maledictus Spiritus.flac",
    r"W:\media\music\[02] organized\[02] lq\Ambient\Tenebrae Aeternae\09 - myNoise - Omen Malum.flac",
    r"W:\media\music\[02] organized\[02] lq\Ambient\Tenebrae Aeternae\10 - myNoise - Inquietus Somnus.flac",
    r"W:\media\music\[02] organized\[02] lq\Ambient\Tenebrae Aeternae\11 - myNoise - Temporis Infinitas.flac",
    r"W:\media\music\[02] organized\[02] lq\Ambient\Tenebrae Aeternae\12 - myNoise - Terribile Monstrum.flac",

]

class EnergyCalculationTest(TestInterface):
    def __init__(self, logger: HoornLogger, main_data_path: Path, mfcc_data_path: Path, train_model: bool = False):
        super().__init__(logger, is_child=True)
        self._logger = logger
        self._separator = self.__class__.__name__

        data_builder = TrainingDataBuilder()
        config_factory = TrainingConfigFactory()

        main_df = pd.read_csv(main_data_path)
        mfcc_df = pd.read_csv(mfcc_data_path)

        self._main_data = data_builder.build(main_df, mfcc_df, DEFAULT_ENERGY_MODEL_CONFIG.use_mfcc)

        self._energy_factory = EnergyFactory(self._logger)
        manager = self._energy_factory.create_lifecycle_manager(Implementation.Default)
        if manager is None:
            raise RuntimeError("Failed to create the EnergyModelLifecycleManager.")

        final_config = config_factory.create(DEFAULT_ENERGY_MODEL_CONFIG, self._main_data)

        if not train_model:
            self._logger.info("Loading existing energy model...", separator=self._separator)

            model_to_use = manager.load_model(final_config)
            if not manager.validate_model(model_to_use, self._main_data):
                self._logger.warning("Loaded model is stale or invalid. Consider retraining.", separator=self._separator)
        else:
            self._logger.info("Training new energy model...", separator=self._separator)
            model_to_use = manager.train_and_persist_model(final_config, self._main_data)

        self._energy_calculator: EnergyCalculator | None = None
        if model_to_use:
            self._logger.info("Creating energy calculator with the configured model.", separator=self._separator)
            self._energy_calculator = self._energy_factory.create_calculator(Implementation.Default, model_to_use)
        else:
            self._logger.error("Could not load or train a valid energy model. Calculator will be unavailable.", separator=self._separator)

        self._calculators: Dict[str, EnergyCalculator] = {
            "Gemini  ": self._energy_calculator
        }

        # Invert the mapping for easy lookup in the test method
        self._header_to_feature_map = {v.value: k for k, v in FEATURE_TO_HEADER_MAPPING.items()}

    def _calculate_and_format_energy(self, row_df: pd.DataFrame) -> str:
        """
        Iterates through the available calculators, computes the energy rating,
        and returns a formatted string of the results.
        """
        energy_strings: List[str] = []
        for header, extractor in self._calculators.items():
            if not extractor:
                continue
            energy_rating: float = extractor.calculate_energy_for_row(row_df)
            energy_strings.append(f"\n- {header}: {energy_rating:.4f}")
        return "".join(energy_strings)

    def test(self) -> None:
        """
        Filters the main data and logs the calculated energy ratings for each track.
        """
        df = self._main_data
        mask = df[Header.Audio_Path.value].isin(TO_CHECK)
        filtered_df = df[mask]

        for index, row in filtered_df.iterrows():
            single_row_df = filtered_df.loc[[index]]

            # Calculate and format the energy ratings
            energy_string = self._calculate_and_format_energy(single_row_df)

            # Log the results
            self._logger.info(
                f"Energy Rating for {row[Header.Title.value]} by {row[Header.Primary_Artist.value]}:{energy_string}",
                separator=self._separator
            )
