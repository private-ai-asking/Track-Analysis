from pathlib import Path
from typing import Dict

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_feature_orchestrator_factory import \
    AudioFeatureOrchestratorFactory
from track_analysis.components.track_analysis.features.data_generation.processors.main_feature_processor import \
    MainFeatureProcessor
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.core.config.default import \
    DEFAULT_KEY_PROGRESSION_CONFIG
from track_analysis.components.track_analysis.library.timing.timing_analysis import TimingAnalyzer
from track_analysis.components.track_analysis.shared.caching.max_rate_cache import \
    MaxRateCache
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext


class CreateProcessors(IPipe):
    def __init__(self, logger: HoornLogger,
                 hop_length: int, n_fft: int,
                 max_rate_cache: MaxRateCache,
                 timing_analyzer: TimingAnalyzer,
                 num_workers: int,
                 number_of_mfccs: int):
        self._separator = "BuildCSV.CreateProcessorsPipe"
        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

        self._hop_length = hop_length
        self._n_fft = n_fft
        self._max_rate_cache = max_rate_cache
        self._num_workers = num_workers
        self._timing_analyzer = timing_analyzer
        self._number_of_mfccs = number_of_mfccs

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        self._logger.trace("Creating processors.", separator=self._separator)

        audio_feature_orchestrator_factory = AudioFeatureOrchestratorFactory(self._logger)
        orchestrator = audio_feature_orchestrator_factory.create_audio_feature_orchestrator(
            hop_length=self._hop_length, n_fft=self._n_fft, max_rate_cache=self._max_rate_cache,
            energy_calculator=data.energy_calculator, key_extraction_config=DEFAULT_KEY_PROGRESSION_CONFIG, timing_analyzer=self._timing_analyzer, number_of_mfccs=self._number_of_mfccs
        )
        main_processor: MainFeatureProcessor = MainFeatureProcessor(
            orchestrator, self._logger,
            cpu_workers=self._num_workers,
            timing_analyzer=self._timing_analyzer
        )

        data.main_processor = main_processor

        return data

    @staticmethod
    def _get_existing_tempo_cache(loaded_audio_info_cache: pd.DataFrame) -> Dict[Path, float]:
        return {
            Path(p): bpm
            for p, bpm in zip(
                loaded_audio_info_cache[Header.Audio_Path.value],
                loaded_audio_info_cache[Header.BPM.value]
            )
        }
