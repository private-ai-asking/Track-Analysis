from pathlib import Path
from typing import Dict

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.testing import TestInterface
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.configuration.default import \
    DEFAULT_ENERGY_MODEL_CONFIG
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.energy_calculator import \
    EnergyCalculator
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.energy_factory import \
    EnergyFactory, Implementation
from track_analysis.components.track_analysis.features.data_generation.model.header import Header

TO_CHECK = [
    r"W:\media\music\[02] organized\[01] hq\OST\Path of Exile (Original Game Soundtrack)\29 Chamber of Innocence.flac",
    r"W:\media\music\[02] organized\[01] hq\OST\Path of Exile (Original Game Soundtrack)\30 High Templar Avarius.flac",
    r"W:\media\music\[02] organized\[01] hq\OST\Path of Exile (Original Game Soundtrack)\08 Merveil.flac",
    r"W:\media\music\[02] organized\[01] hq\Reggae\Damian Marley\Stony Hill (Explicit)\03 Damian “Jr. Gong” Marley - Nail Pon Cross.flac",
    r"W:\media\music\[02] organized\[02] lq\Ambient\Tenebrae Aeternae\01 - myNoise - Ora Pro Nobis.flac",
    r"W:\media\music\[02] organized\[02] lq\Ambient\Tenebrae Aeternae\07 - myNoise - Homini Lupus.flac",
    r"W:\media\music\[02] organized\[01] hq\Classical\Ludovico Einaudi\Elegy For The Arctic\01 Elegy for the Arctic.flac"
]

class EnergyCalculationTest(TestInterface):
    def __init__(self, logger: HoornLogger, main_data_path: Path, mfcc_data_path: Path, train_model: bool = False):
        super().__init__(logger, is_child=True)
        self._logger = logger
        self._separator = self.__class__.__name__
        self._main_data = pd.read_csv(main_data_path)
        self._mfcc_data = pd.read_csv(mfcc_data_path)

        self._energy_factory = EnergyFactory(self._logger, self._mfcc_data)
        manager = self._energy_factory.create_lifecycle_manager(Implementation.Default)
        if manager is None:
            raise RuntimeError("Failed to create the EnergyModelLifecycleManager.")

        if not train_model:
            self._logger.info("Loading existing energy model...", separator=self._separator)
            model = manager.load_model(DEFAULT_ENERGY_MODEL_CONFIG)
            if not manager.validate_model(model, self._main_data):
                self._logger.warning("Loaded model is stale or invalid. Consider retraining.", separator=self._separator)
        else:
            self._logger.info("Training new energy model...", separator=self._separator)
            model = manager.train_and_persist_model(DEFAULT_ENERGY_MODEL_CONFIG, self._main_data)

        self._energy_calculator: EnergyCalculator | None = None
        if model:
            self._logger.info("Creating energy calculator with the configured model.", separator=self._separator)
            self._energy_calculator = self._energy_factory.create_calculator(Implementation.Default, model)
        else:
            self._logger.error("Could not load or train a valid energy model. Calculator will be unavailable.", separator=self._separator)

        self._calculators: Dict[str, EnergyCalculator] = {
            "Gemini  ": self._energy_calculator
        }

    def test(self) -> None:
        df = self._main_data
        mask = df[Header.Audio_Path.value].isin(TO_CHECK)
        filtered = df[mask]
        for _, row in filtered.iterrows():
            energy_string: str = ""
            for header, extractor in self._calculators.items():
                energy_rating: float = extractor.calculate_energy_for_row(row)
                energy_string += f"\n- {header}: {energy_rating:.4f}"

            self._logger.info(f"Energy Rating for {row[Header.Title.value]} by {row[Header.Primary_Artist.value]}:{energy_string}", separator=self._separator)
