from pathlib import Path
from typing import Dict

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.testing import TestInterface
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.default_predictor import \
    DefaultAudioEnergyPredictor
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.trainer import \
    DefaultEnergyModelTrainer
from track_analysis.components.track_analysis.features.data_generation.energy_calculation.energy_calculator import \
    EnergyCalculator
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
    def __init__(self, logger: HoornLogger, cache_path: Path):
        super().__init__(logger, is_child=True)
        self._logger = logger
        self._separator = self.__class__.__name__
        self._cache = pd.read_csv(cache_path)

        trainer = DefaultEnergyModelTrainer(self._logger)
        model = trainer.train_or_load(self._cache)
        predictor = DefaultAudioEnergyPredictor(self._logger, model)

        self._calculators: Dict[str, EnergyCalculator] = {
            "Gemini  ": predictor
        }

    def test(self) -> None:
        df = self._cache
        mask = df[Header.Audio_Path.value].isin(TO_CHECK)
        filtered = df[mask]
        for _, row in filtered.iterrows():
            energy_string: str = ""
            for header, extractor in self._calculators.items():
                energy_rating: float = extractor.calculate_energy_for_row(row)
                energy_string += f"\n- {header}: {energy_rating:.4f}"

            self._logger.info(f"Energy Rating for {row[Header.Title.value]} by {row[Header.Primary_Artist.value]}:{energy_string}", separator=self._separator)
