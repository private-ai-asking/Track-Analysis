from typing import List, Dict, Any

import numpy as np

from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_calculator import \
    AudioDataFeatureCalculator
from track_analysis.components.track_analysis.features.core.cacheing.multi_band_onset import OnsetStrengthMultiExtractor


class MultiBandOnsetEnvelopeCalculator(AudioDataFeatureCalculator):
    """
    Intermediate calculator for multi-band onset strength envelopes.
    Accepts a shared OnsetStrengthMultiExtractor instance to avoid redundant computation.
    """
    def __init__(self, onset_multi_extractor: OnsetStrengthMultiExtractor):
        self._onset_multi_extractor = onset_multi_extractor

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.AUDIO_PATH,
            AudioDataFeature.AUDIO_SAMPLE_RATE,
            AudioDataFeature.PERCUSSIVE_AUDIO,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        percussive = data[AudioDataFeature.PERCUSSIVE_AUDIO]
        common_args = {
            "file_path": data[AudioDataFeature.AUDIO_PATH],
            "sample_rate": data[AudioDataFeature.AUDIO_SAMPLE_RATE],
            "audio": percussive,
            "start_sample": 0,
            "end_sample": len(percussive),
            "hop_length": 512,
        }
        return {
            AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES: self._onset_multi_extractor.extract(**common_args)
        }


class MultiBandOnsetPeaksCalculator(AudioDataFeatureCalculator):
    """
    Intermediate calculator for multi-band onset peaks.
    Accepts a shared OnsetStrengthMultiExtractor instance to avoid redundant computation.
    """
    def __init__(self, onset_multi_extractor: OnsetStrengthMultiExtractor):
        self._onset_multi_extractor = onset_multi_extractor

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.AUDIO_PATH,
            AudioDataFeature.AUDIO_SAMPLE_RATE,
            AudioDataFeature.PERCUSSIVE_AUDIO,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.MULTI_BAND_ONSET_PEAKS

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        percussive = data[AudioDataFeature.PERCUSSIVE_AUDIO]
        common_args = {
            "file_path": data[AudioDataFeature.AUDIO_PATH],
            "sample_rate": data[AudioDataFeature.AUDIO_SAMPLE_RATE],
            "audio": percussive,
            "start_sample": 0,
            "end_sample": len(percussive),
            "hop_length": 512,
        }
        return {
            AudioDataFeature.MULTI_BAND_ONSET_PEAKS: self._onset_multi_extractor.extract_peaks(**common_args)
        }

# --- KICK Calculators ---

class OnsetEnvMeanKickCalculator(AudioDataFeatureCalculator):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_ENV_MEAN_KICK

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        onset_envs = data[AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES]
        return {AudioDataFeature.ONSET_ENV_MEAN_KICK: float(onset_envs.get("kick", np.array([0.0])).mean())}


class OnsetRateKickCalculator(AudioDataFeatureCalculator):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.MULTI_BAND_ONSET_PEAKS,
            AudioDataFeature.AUDIO_SAMPLES,
            AudioDataFeature.AUDIO_SAMPLE_RATE,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_RATE_KICK

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        onset_peaks = data[AudioDataFeature.MULTI_BAND_ONSET_PEAKS]
        samples = data[AudioDataFeature.AUDIO_SAMPLES]
        sr = data[AudioDataFeature.AUDIO_SAMPLE_RATE]
        duration_sec = len(samples) / sr if sr > 0 else 1.0
        return {AudioDataFeature.ONSET_RATE_KICK: len(onset_peaks.get("kick", [])) / duration_sec}


# --- SNARE Calculators ---

class OnsetEnvMeanSnareCalculator(AudioDataFeatureCalculator):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_ENV_MEAN_SNARE

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        onset_envs = data[AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES]
        return {AudioDataFeature.ONSET_ENV_MEAN_SNARE: float(onset_envs.get("snare", np.array([0.0])).mean())}


class OnsetRateSnareCalculator(AudioDataFeatureCalculator):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.MULTI_BAND_ONSET_PEAKS,
            AudioDataFeature.AUDIO_SAMPLES,
            AudioDataFeature.AUDIO_SAMPLE_RATE,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_RATE_SNARE

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        onset_peaks = data[AudioDataFeature.MULTI_BAND_ONSET_PEAKS]
        samples = data[AudioDataFeature.AUDIO_SAMPLES]
        sr = data[AudioDataFeature.AUDIO_SAMPLE_RATE]
        duration_sec = len(samples) / sr if sr > 0 else 1.0
        return {AudioDataFeature.ONSET_RATE_SNARE: len(onset_peaks.get("snare", [])) / duration_sec}


# --- LOW-MID Calculators ---

class OnsetEnvMeanLowMidCalculator(AudioDataFeatureCalculator):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_ENV_MEAN_LOW_MID

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        onset_envs = data[AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES]
        return {AudioDataFeature.ONSET_ENV_MEAN_LOW_MID: float(onset_envs.get("low_mid", np.array([0.0])).mean())}


class OnsetRateLowMidCalculator(AudioDataFeatureCalculator):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.MULTI_BAND_ONSET_PEAKS,
            AudioDataFeature.AUDIO_SAMPLES,
            AudioDataFeature.AUDIO_SAMPLE_RATE,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_RATE_LOW_MID

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        onset_peaks = data[AudioDataFeature.MULTI_BAND_ONSET_PEAKS]
        samples = data[AudioDataFeature.AUDIO_SAMPLES]
        sr = data[AudioDataFeature.AUDIO_SAMPLE_RATE]
        duration_sec = len(samples) / sr if sr > 0 else 1.0
        return {AudioDataFeature.ONSET_RATE_LOW_MID: len(onset_peaks.get("low_mid", [])) / duration_sec}


# --- HI-HAT Calculators ---

class OnsetEnvMeanHiHatCalculator(AudioDataFeatureCalculator):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_ENV_MEAN_HI_HAT

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        onset_envs = data[AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES]
        return {AudioDataFeature.ONSET_ENV_MEAN_HI_HAT: float(onset_envs.get("hihat", np.array([0.0])).mean())}


class OnsetRateHiHatCalculator(AudioDataFeatureCalculator):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.MULTI_BAND_ONSET_PEAKS,
            AudioDataFeature.AUDIO_SAMPLES,
            AudioDataFeature.AUDIO_SAMPLE_RATE,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_RATE_HI_HAT

    def calculate(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        onset_peaks = data[AudioDataFeature.MULTI_BAND_ONSET_PEAKS]
        samples = data[AudioDataFeature.AUDIO_SAMPLES]
        sr = data[AudioDataFeature.AUDIO_SAMPLE_RATE]
        duration_sec = len(samples) / sr if sr > 0 else 1.0
        return {AudioDataFeature.ONSET_RATE_HI_HAT: len(onset_peaks.get("hihat", [])) / duration_sec}
