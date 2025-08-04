from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_calculator_orchestrator import \
    AudioDataFeatureCalculatorOrchestrator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.loudness.crest_factor import \
    CrestFactorCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.loudness.integrated_lufs import \
    IntegratedLufsCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.loudness.loudness_analyzer import \
    LoudnessAnalyzer
from track_analysis.components.track_analysis.features.audio_calculation.calculators.loudness.loudness_range import \
    LoudnessRangeCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.loudness.true_peak import \
    TruePeakCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.misc.hpss import HPSSExtractor
from track_analysis.components.track_analysis.features.audio_calculation.calculators.rms.iqr_rms_calculator import \
    IQRRmsCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.rms.max_rms_calculator import \
    MaxRmsCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.rms.mean_rms_calculator import \
    MeanRmsCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.rms.percentile_rms_calculator import \
    PercentileRmsCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.centroid import \
    SpectralCentroidCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.centroid_and_flux import \
    SpectralCentroidAndFluxCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.contrast import \
    SpectralContrastCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.dynamic_tempo import \
    DynamicTempoCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.flatness import \
    SpectralFlatnessCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.flux import \
    SpectralFluxCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.harmonicity import \
    HarmonicityCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.magnitude import \
    HarmonicSpectrogramCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.mfcc import MfccCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.multi_band_onset import \
    MultiBandOnsetPeaksCalculator, MultiBandOnsetEnvelopeCalculator, OnsetEnvMeanKickCalculator, \
    OnsetEnvMeanSnareCalculator, OnsetEnvMeanLowMidCalculator, OnsetEnvMeanHiHatCalculator, OnsetRateKickCalculator, \
    OnsetRateSnareCalculator, OnsetRateLowMidCalculator, OnsetRateHiHatCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.onset_env_mean import \
    OnsetEnvMeanCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.onset_envelope import \
    OnsetEnvelopeCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.onset_peaks import \
    OnsetPeaksCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.onset_rate import \
    OnsetRateCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.rolloff import \
    SpectralRolloffCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.tempo_variation import \
    TempoVariationCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.zcr import ZCRCalculator
from track_analysis.components.track_analysis.features.core.cacheing.magnitude_spectogram import \
    MagnitudeSpectrogramExtractor
from track_analysis.components.track_analysis.features.core.cacheing.multi_band_onset import OnsetStrengthMultiExtractor


class AudioFeatureOrchestratorFactory:
    def __init__(self, logger: HoornLogger):
        self._logger = logger

    def create_audio_feature_orchestrator(self, hop_length: int = 512, n_fft: int = 2048) -> AudioDataFeatureCalculatorOrchestrator:
        """Factory function to assemble and configure all audio feature calculators."""
        magnitude_extractor = MagnitudeSpectrogramExtractor(self._logger, n_fft=n_fft, hop_length=hop_length)
        onset_multi_extractor = OnsetStrengthMultiExtractor(self._logger, magnitude_extractor)

        all_calculators = [
            CrestFactorCalculator(), IntegratedLufsCalculator(), LoudnessAnalyzer(),
            LoudnessRangeCalculator(), TruePeakCalculator(), HPSSExtractor(self._logger, hop_length=hop_length, n_fft=n_fft),
            IQRRmsCalculator(), MaxRmsCalculator(), MeanRmsCalculator(), PercentileRmsCalculator(),
            SpectralCentroidCalculator(), SpectralCentroidAndFluxCalculator(),
            SpectralContrastCalculator(self._logger, hop_length=hop_length),
            SpectralFlatnessCalculator(self._logger, hop_length=hop_length), SpectralFluxCalculator(),
            HarmonicityCalculator(self._logger), MfccCalculator(self._logger),
            MultiBandOnsetPeaksCalculator(onset_multi_extractor), MultiBandOnsetEnvelopeCalculator(onset_multi_extractor),
            OnsetEnvMeanCalculator(), OnsetEnvMeanKickCalculator(), OnsetEnvMeanSnareCalculator(),
            OnsetEnvMeanLowMidCalculator(), OnsetEnvMeanHiHatCalculator(), OnsetRateCalculator(),
            OnsetRateKickCalculator(), OnsetRateSnareCalculator(), OnsetRateLowMidCalculator(),
            OnsetRateHiHatCalculator(), SpectralRolloffCalculator(self._logger, hop_length=hop_length),
            TempoVariationCalculator(), ZCRCalculator(self._logger, hop_length=hop_length),
            HarmonicSpectrogramCalculator(self._logger, hop_length=hop_length, n_fft=n_fft),
            OnsetEnvelopeCalculator(self._logger, hop_length=hop_length),
            OnsetPeaksCalculator(self._logger, hop_length=hop_length),
            DynamicTempoCalculator(self._logger, hop_length=hop_length),
        ]

        return AudioDataFeatureCalculatorOrchestrator(all_calculators)
