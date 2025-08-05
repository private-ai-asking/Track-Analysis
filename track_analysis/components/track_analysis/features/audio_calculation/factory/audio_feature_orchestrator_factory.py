from pathlib import Path
from typing import Dict

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_provider_orchestrator import \
    AudioDataFeatureProviderOrchestrator
from track_analysis.components.track_analysis.features.audio_calculation.providers.basic.audio_data_provider import \
    AudioDataProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.basic.data_efficiency_provider import \
    DataEfficiencyProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.basic.raw_audio_info_provider import \
    RawAudioInfoProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.basic.tempo_provider import \
    TempoProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.loudness.crest_factor import \
    CrestFactorProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.loudness.integrated_lufs import \
    IntegratedLufsProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.loudness.loudness_analyzer import \
    LoudnessAnalyzer
from track_analysis.components.track_analysis.features.audio_calculation.providers.loudness.loudness_range import \
    LoudnessRangeProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.loudness.true_peak import \
    TruePeakProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.misc.hpss import HPSSExtractor
from track_analysis.components.track_analysis.features.audio_calculation.providers.rms.iqr_rms import \
    IQRRmsProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.rms.max_rms import \
    MaxRmsProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.rms.mean_rms import \
    MeanRmsProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.rms.percentile_rms import \
    PercentileRmsProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.spectral.centroid import \
    SpectralCentroidProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.spectral.centroid_and_flux import \
    SpectralCentroidAndFluxProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.spectral.contrast import \
    SpectralContrastProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.spectral.dynamic_tempo import \
    DynamicTempoProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.spectral.flatness import \
    SpectralFlatnessProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.spectral.flux import \
    SpectralFluxProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.spectral.harmonicity import \
    HarmonicityProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.spectral.magnitude import \
    HarmonicSpectrogramProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.spectral.mfcc import MfccProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.spectral.multi_band_onset import \
    MultiBandOnsetPeaksProvider, MultiBandOnsetEnvelopeProvider, OnsetEnvMeanKickProvider, \
    OnsetEnvMeanSnareProvider, OnsetEnvMeanLowMidProvider, OnsetEnvMeanHiHatProvider, OnsetRateKickProvider, \
    OnsetRateSnareProvider, OnsetRateLowMidProvider, OnsetRateHiHatProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.spectral.onset_env_mean import \
    OnsetEnvMeanProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.spectral.onset_envelope import \
    OnsetEnvelopeProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.spectral.onset_peaks import \
    OnsetPeaksProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.spectral.onset_rate import \
    OnsetRateProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.spectral.rolloff import \
    SpectralRolloffProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.spectral.tempo_variation import \
    TempoVariationProvider
from track_analysis.components.track_analysis.features.audio_calculation.providers.spectral.zcr import ZCRProvider
from track_analysis.components.track_analysis.features.audio_calculation.utils.cacheing.max_rate_cache import \
    MaxRateCache
from track_analysis.components.track_analysis.features.audio_calculation.utils.file_utils import FileUtils
from track_analysis.components.track_analysis.features.core.cacheing.beat import BeatDetector
from track_analysis.components.track_analysis.features.core.cacheing.magnitude_spectogram import \
    MagnitudeSpectrogramExtractor
from track_analysis.components.track_analysis.features.core.cacheing.multi_band_onset import OnsetStrengthMultiExtractor


class AudioFeatureOrchestratorFactory:
    def __init__(self, logger: HoornLogger):
        self._logger = logger

    def create_audio_feature_orchestrator(self,
                                          max_rate_cache: MaxRateCache,
                                          existing_tempo_cache: Dict[Path, float] | None = None,
                                          hop_length: int = 512,
                                          n_fft: int = 2048,
                                          ) -> AudioDataFeatureProviderOrchestrator:
        """Factory function to assemble and configure all audio feature calculators."""
        magnitude_extractor = MagnitudeSpectrogramExtractor(self._logger, n_fft=n_fft, hop_length=hop_length)
        onset_multi_extractor = OnsetStrengthMultiExtractor(self._logger, magnitude_extractor)
        beat_detector: BeatDetector = BeatDetector(self._logger, existing_tempo_cache=existing_tempo_cache)
        file_utils: FileUtils = FileUtils()

        all_calculators = [
            CrestFactorProvider(), IntegratedLufsProvider(), LoudnessAnalyzer(),
            LoudnessRangeProvider(), TruePeakProvider(), HPSSExtractor(self._logger, hop_length=hop_length, n_fft=n_fft),
            IQRRmsProvider(), MaxRmsProvider(), MeanRmsProvider(), PercentileRmsProvider(),
            SpectralCentroidProvider(), SpectralCentroidAndFluxProvider(),
            SpectralContrastProvider(self._logger, hop_length=hop_length),
            SpectralFlatnessProvider(self._logger, hop_length=hop_length), SpectralFluxProvider(),
            HarmonicityProvider(self._logger), MfccProvider(self._logger),
            MultiBandOnsetPeaksProvider(onset_multi_extractor), MultiBandOnsetEnvelopeProvider(onset_multi_extractor),
            OnsetEnvMeanProvider(), OnsetEnvMeanKickProvider(), OnsetEnvMeanSnareProvider(),
            OnsetEnvMeanLowMidProvider(), OnsetEnvMeanHiHatProvider(), OnsetRateProvider(),
            OnsetRateKickProvider(), OnsetRateSnareProvider(), OnsetRateLowMidProvider(),
            OnsetRateHiHatProvider(), SpectralRolloffProvider(self._logger, hop_length=hop_length),
            TempoVariationProvider(), ZCRProvider(self._logger, hop_length=hop_length),
            HarmonicSpectrogramProvider(self._logger, hop_length=hop_length, n_fft=n_fft),
            OnsetEnvelopeProvider(self._logger, hop_length=hop_length),
            OnsetPeaksProvider(self._logger, hop_length=hop_length),
            DynamicTempoProvider(self._logger, hop_length=hop_length),
            AudioDataProvider(self._logger), RawAudioInfoProvider(self._logger),
            TempoProvider(beat_detector, hop_length=hop_length),
            DataEfficiencyProvider(file_utils, max_rate_cache)
        ]

        return AudioDataFeatureProviderOrchestrator(all_calculators, self._logger)
