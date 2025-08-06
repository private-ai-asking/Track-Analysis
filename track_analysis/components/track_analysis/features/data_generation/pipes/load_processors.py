from pathlib import Path
from typing import Dict

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_feature_orchestrator_factory import \
    AudioFeatureOrchestratorFactory
from track_analysis.components.track_analysis.features.data_generation.processors.key_feature_processor import \
    KeyFeatureProcessor
from track_analysis.components.track_analysis.features.data_generation.processors.main_feature_processor import \
    MainFeatureProcessor
from track_analysis.components.track_analysis.shared.caching.max_rate_cache import \
    MaxRateCache
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.data_generation.helpers.key_extractor import KeyExtractor


class CreateProcessors(IPipe):
    def __init__(self, logger: HoornLogger,
                 hop_length: int, n_fft: int,
                 max_rate_cache: MaxRateCache, key_extractor: KeyExtractor,
                 num_workers: int):
        self._separator = "BuildCSV.CreateProcessorsPipe"
        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

        self._hop_length = hop_length
        self._n_fft = n_fft
        self._max_rate_cache = max_rate_cache
        self._num_workers = num_workers
        self._key_extractor = key_extractor

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        self._logger.trace("Creating processors.", separator=self._separator)

        audio_feature_orchestrator_factory = AudioFeatureOrchestratorFactory(self._logger)
        orchestrator = audio_feature_orchestrator_factory.create_audio_feature_orchestrator(
            hop_length=self._hop_length, n_fft=self._n_fft, max_rate_cache=self._max_rate_cache, existing_tempo_cache=self._get_existing_tempo_cache(data.loaded_audio_info_cache),
            energy_calculator=data.energy_calculator
        )
        main_processor: MainFeatureProcessor = MainFeatureProcessor(
            orchestrator, self._logger,
            max_io_workers=data.max_new_tracks_per_run,
            cpu_workers=self._num_workers,
            adjustment_interval=25
        )
        key_processor = KeyFeatureProcessor(self._key_extractor, self._logger)

        data.main_processor = main_processor
        data.key_processor = key_processor

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
