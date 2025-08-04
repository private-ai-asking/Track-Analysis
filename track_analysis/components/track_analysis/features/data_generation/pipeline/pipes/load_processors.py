from pathlib import Path
from typing import List, Dict

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.factory.audio_feature_orchestrator_factory import \
    AudioFeatureOrchestratorFactory
from track_analysis.components.track_analysis.features.audio_calculation.feature_to_header_mapping import \
    FEATURE_TO_HEADER_MAPPING
from track_analysis.components.track_analysis.features.audio_calculation.processors.key_feature_processor import \
    KeyFeatureProcessor
from track_analysis.components.track_analysis.features.audio_calculation.processors.sample_feature_processor import \
    SampleFeatureProcessor
from track_analysis.components.track_analysis.features.audio_calculation.utils.cacheing.max_rate_cache import \
    MaxRateCache
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.data_generation.util.key_extractor import KeyExtractor


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

        to_calculate: List[AudioDataFeature] = list(FEATURE_TO_HEADER_MAPPING.keys())
        to_calculate.extend([AudioDataFeature.MFCC_MEANS, AudioDataFeature.MFCC_STDS])

        audio_feature_orchestrator_factory = AudioFeatureOrchestratorFactory(self._logger)
        orchestrator = audio_feature_orchestrator_factory.create_audio_feature_orchestrator(hop_length=self._hop_length, n_fft=self._n_fft, max_rate_cache=self._max_rate_cache, existing_tempo_cache=self._get_existing_tempo_cache(data.loaded_audio_info_cache))
        sample_processor: SampleFeatureProcessor = SampleFeatureProcessor(orchestrator, to_calculate, self._logger, num_workers=self._num_workers)
        key_processor = KeyFeatureProcessor(self._key_extractor, self._logger)

        data.sample_processor = sample_processor
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
