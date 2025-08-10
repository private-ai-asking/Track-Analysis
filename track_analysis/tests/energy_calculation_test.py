from pathlib import Path
from typing import Dict, List

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.testing import TestInterface
from track_analysis.components.track_analysis.features.data_generation.helpers.energy_loader import EnergyLoader
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.energy_calculator import \
    EnergyAlgorithm
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.feature_to_header_mapping import \
    FEATURE_TO_HEADER_MAPPING
from track_analysis.components.track_analysis.library.configuration.model.configuration import \
    TrackAnalysisConfigurationModel

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
    def __init__(self,
                 logger: HoornLogger,
                 track_analysis_config: TrackAnalysisConfigurationModel
                 ):
        super().__init__(logger, is_child=True)
        self._logger = logger
        self._separator = self.__class__.__name__

        self._main_df = pd.read_csv(track_analysis_config.paths.library_data)
        self._mfcc_df = pd.read_csv(track_analysis_config.paths.mfcc_data)

        self._energy_loader: EnergyLoader = EnergyLoader(logger,
                                                         energy_training_version_to_use=track_analysis_config.additional_config.current_energy_training_version_to_use,
                                                         regenerate_library_growth_threshold=track_analysis_config.additional_config.energy_calculation_regenerate_library_growth_perc,
                                                         cache_dir=track_analysis_config.paths.cache_dir)
        self._combined_data = self._energy_loader.get_combined_data(self._main_df, self._mfcc_df)

        # Invert the mapping for easy lookup in the test method
        self._header_to_feature_map = {v.value: k for k, v in FEATURE_TO_HEADER_MAPPING.items()}

    def _calculate_and_format_energy(self, row_df: pd.DataFrame, calculators: Dict[str, EnergyAlgorithm]) -> str:
        """
        Iterates through the available calculators, computes the energy rating,
        and returns a formatted string of the results.
        """
        energy_strings: List[str] = []
        for header, extractor in calculators.items():
            if not extractor:
                continue
            energy_rating: float = extractor.calculate_energy_for_row(row_df)
            energy_strings.append(f"\n- {header}: {energy_rating:.4f}")
        return "".join(energy_strings)

    def test(self) -> None:
        """
        Filters the main data and logs the calculated energy ratings for each track.
        """
        df = self._combined_data
        mask = df[Header.Audio_Path.value].isin(TO_CHECK)
        filtered_df = df[mask]

        calculators: Dict[str, EnergyAlgorithm] = {
            "Gemini   ": self._energy_loader.get_calculator(self._main_df, self._mfcc_df)
        }

        for index, row in filtered_df.iterrows():
            single_row_df = filtered_df.loc[[index]]

            # Calculate and format the energy ratings
            energy_string = self._calculate_and_format_energy(single_row_df, calculators)

            # Log the results
            self._logger.info(
                f"Energy Rating for {row[Header.Title.value]} by {row[Header.Primary_Artist.value]}:{energy_string}",
                separator=self._separator
            )
