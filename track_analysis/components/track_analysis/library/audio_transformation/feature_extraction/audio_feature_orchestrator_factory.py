from pathlib import Path
from typing import Dict

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.energy_calculation.energy_calculator import \
    EnergyAlgorithm
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider_orchestrator import \
    AudioDataFeatureProviderOrchestrator
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.basic.data_efficiency_provider import \
    DataEfficiencyProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.chroma.chroma_entropy import \
    ChromaEntropyProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.chroma.chromagram import \
    ChromagramProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.energy.energy_provider import \
    EnergyProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.energy.multi_band_energy import \
    MultiBandEnergyProvider, LowMidEnergyRatioProvider, SubBassEnergyRatioProvider, BassEnergyRatioProvider, \
    MidEnergyRatioProvider, HighEnergyRatioProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.loudness.crest_factor import \
    CrestFactorProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.loudness.integrated_lufs import \
    IntegratedLufsProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.loudness.intermediary.loudness_analyzer import \
    LoudnessAnalyzer
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.loudness.loudness_range import \
    LoudnessRangeProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.loudness.rms.iqr_rms import \
    IQRRmsProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.loudness.rms.max_rms import \
    MaxRmsProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.loudness.rms.mean_rms import \
    MeanRmsProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.loudness.rms.percentile_rms import \
    PercentileRmsProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.loudness.true_peak import \
    TruePeakProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.beat_frames import \
    BeatFramesProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.beat_strength import \
    BeatStrengthProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.calculator.multi_band_onset import \
    OnsetStrengthMultiExtractor
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.dynamic_tempo import \
    DynamicTempoProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.onset.multi_band_onset import \
    OnsetRateHiHatProvider, OnsetRateLowMidProvider, OnsetRateSnareProvider, OnsetRateKickProvider, \
    OnsetEnvMeanHiHatProvider, OnsetEnvMeanLowMidProvider, OnsetEnvMeanSnareProvider, OnsetEnvMeanKickProvider, \
    MultiBandOnsetEnvelopeProvider, MultiBandOnsetPeaksProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.onset.onset_env_mean import \
    OnsetEnvMeanProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.onset.onset_envelope import \
    OnsetEnvelopeProvider, PercussiveOnsetEnvelopeProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.onset.onset_peaks import \
    OnsetPeaksProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.onset.onset_peaks_percussive import \
    PercussiveOnsetPeaksProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.onset.onset_rate import \
    OnsetRateProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.onset.onset_variation_provider import \
    OnsetVariationProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.rhythmic_regularity import \
    RhythmicRegularityProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.tempo_provider import \
    BeatDetector, TempoProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.tempo_variation import \
    TempoVariationProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.zcr import \
    ZCRProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.separation.hpss import \
    HPSSExtractor
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.calculator.magnitude_spectogram import \
    MagnitudeSpectrogramExtractor
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.centroid import \
    SpectralCentroidProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.centroid_and_flux import \
    SpectralCentroidAndFluxProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.contrast import \
    SpectralContrastProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.flatness import \
    SpectralFlatnessProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.flux import \
    SpectralFluxProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.frequencies import \
    FFTFrequenciesProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.harmonic_to_percussive_ratio import \
    HarmonicToPercussiveRatioProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.harmonicity import \
    HarmonicityProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.magnitude import \
    HarmonicSpectrogramProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.mfcc import \
    MfccProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.rolloff import \
    SpectralRolloffProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.spectral_bandwidth import \
    SpectralBandwidthProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.spectral_entropy import \
    SpectralEntropyProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.spectral.spectral_skewness_and_kurtosis import \
    SpectralMomentsProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.raw.audio_sample_provider import \
    AudioSampleProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.raw.raw_audio_info_provider import \
    RawAudioInfoProvider
from track_analysis.components.track_analysis.shared.caching.max_rate_cache import MaxRateCache
from track_analysis.components.track_analysis.shared.file_utils import FileUtils


class AudioFeatureOrchestratorFactory:
    def __init__(self, logger: HoornLogger):
        self._logger = logger

    def create_audio_feature_orchestrator(self,
                                          max_rate_cache: MaxRateCache,
                                          energy_calculator: EnergyAlgorithm,
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
            CrestFactorProvider(), IntegratedLufsProvider(), LoudnessAnalyzer(hop_size=hop_length),
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
            AudioSampleProvider(self._logger), RawAudioInfoProvider(self._logger),
            TempoProvider(beat_detector, hop_length=hop_length),
            DataEfficiencyProvider(file_utils, max_rate_cache),
            EnergyProvider(energy_calculator, self._logger),
            FFTFrequenciesProvider(n_fft=n_fft),
            MultiBandEnergyProvider(), LowMidEnergyRatioProvider(), SubBassEnergyRatioProvider(), BassEnergyRatioProvider(), MidEnergyRatioProvider(), HighEnergyRatioProvider(),
            HarmonicToPercussiveRatioProvider(),
            PercussiveOnsetPeaksProvider(self._logger, hop_length=hop_length), OnsetVariationProvider(), SpectralMomentsProvider(), SpectralEntropyProvider(),
            BeatFramesProvider(hop_length=hop_length, beat_detector=beat_detector), BeatStrengthProvider(),
            ChromagramProvider(hop_length=hop_length), ChromaEntropyProvider(),
            RhythmicRegularityProvider(), PercussiveOnsetEnvelopeProvider(self._logger, hop_length=hop_length),
            SpectralBandwidthProvider(),
        ]

        return AudioDataFeatureProviderOrchestrator(all_calculators, self._logger)
