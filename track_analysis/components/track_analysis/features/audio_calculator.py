from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import EXPENSIVE_CACHE_DIRECTORY
from track_analysis.components.track_analysis.features.audio_calculation.audio_processing_pipeline import \
    AudioProcessingPipeline, AudioProcessingResult
from track_analysis.components.track_analysis.features.audio_calculation.batch_file_metrics_service import \
    BatchFileMetricsService
from track_analysis.components.track_analysis.features.audio_calculation.batch_sample_metrics_service import \
    BatchSampleMetricsService
from track_analysis.components.track_analysis.features.audio_calculation.calculators.loudness_calculator import \
    LoudnessCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.rms_calculator import RmsCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral_rhythm_calculator import \
    SpectralRhythmCalculator, FeatureExtractors
from track_analysis.components.track_analysis.features.audio_calculation.key_tagging_service import KeyTaggingService
from track_analysis.components.track_analysis.features.audio_calculation.utils.cacheing.max_rate_cache import \
    MaxRateCache
from track_analysis.components.track_analysis.features.audio_calculation.utils.file_utils import FileUtils
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler, AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.core.cacheing.harmonic import HarmonicExtractor
from track_analysis.components.track_analysis.features.core.cacheing.harmonicity import HarmonicityExtractor
from track_analysis.components.track_analysis.features.core.cacheing.magnitude_spectogram import \
    MagnitudeSpectrogramExtractor
from track_analysis.components.track_analysis.features.core.cacheing.mfcc import MfccExtractor
from track_analysis.components.track_analysis.features.core.cacheing.multi_band_onset import OnsetStrengthMultiExtractor
from track_analysis.components.track_analysis.features.core.cacheing.onset_envelope import OnsetStrengthExtractor
from track_analysis.components.track_analysis.features.core.cacheing.spectral_contrast import SpectralContrastExtractor
from track_analysis.components.track_analysis.features.core.cacheing.spectral_flatness import SpectralFlatnessExtractor
from track_analysis.components.track_analysis.features.core.cacheing.spectral_rolloff import SpectralRolloffExtractor
from track_analysis.components.track_analysis.features.core.cacheing.zero_crossing import ZeroCrossingRateExtractor
from track_analysis.components.track_analysis.features.data_generation.util.key_extractor import KeyExtractor


class AudioCalculator:
    def __init__(self,
                 logger: HoornLogger,
                 audio_file_handler: AudioFileHandler,
                 key_csv_path: Path,
                 num_workers: int):

        # metric calculators already wired up elsewhere:
        harmonic_extractor = HarmonicExtractor(logger)
        magnitude_extractor = MagnitudeSpectrogramExtractor(logger)

        onset_extractor = OnsetStrengthExtractor(logger)
        onset_extractor_multi = OnsetStrengthMultiExtractor(logger, magnitude_extractor=magnitude_extractor)
        zcr_extractor = ZeroCrossingRateExtractor(logger)
        flatness_extractor = SpectralFlatnessExtractor(logger)
        contrast_extractor = SpectralContrastExtractor(logger)
        rolloff = SpectralRolloffExtractor(logger)
        harmonicity = HarmonicityExtractor(logger)
        mfcc = MfccExtractor(logger)

        extractors: FeatureExtractors = FeatureExtractors(
            harmonic=harmonic_extractor,
            magnitude=magnitude_extractor,
            onset_global=onset_extractor,
            onset_multi=onset_extractor_multi,
            zcr=zcr_extractor,
            flatness=flatness_extractor,
            contrast=contrast_extractor,
            rolloff=rolloff,
            harmonicity=harmonicity,
            mfcc=mfcc,
        )
        calculators = [LoudnessCalculator(), RmsCalculator(), SpectralRhythmCalculator(extractors, hop_length=512)]
        sample_svc = BatchSampleMetricsService(calculators, num_workers)

        file_utils = FileUtils()
        rate_cache = MaxRateCache(EXPENSIVE_CACHE_DIRECTORY / "max_rate_cache.pkl")
        key_tagger = KeyTaggingService(KeyExtractor(logger, audio_file_handler, num_workers),
                                       key_csv_path,
                                       logger)
        file_svc = BatchFileMetricsService(file_utils, rate_cache, key_tagger)

        self._pipeline = AudioProcessingPipeline(sample_svc, file_svc)
        self._logger = logger

    def process(self,
                infos: List[AudioStreamsInfoModel],
                paths: List[Path],
                uuids: List[str],
                samples: List[np.ndarray],
                sample_rates: List[int],
                tempos: List[int]) -> AudioProcessingResult:
        return self._pipeline.run(infos, paths, uuids, samples, sample_rates, tempos)
