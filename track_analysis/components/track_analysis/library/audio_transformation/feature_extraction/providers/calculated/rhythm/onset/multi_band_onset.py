from typing import List, Dict, Any

import numpy as np

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.calculator.multi_band_onset import \
    OnsetStrengthMultiExtractor


class MultiBandOnsetEnvelopeProvider(AudioDataFeatureProvider):
    """
    Intermediate calculator for multi-band onset strength envelopes.
    Accepts a shared OnsetStrengthMultiExtractor instance to avoid redundant computation.
    """
    def __init__(self, onset_multi_extractor: OnsetStrengthMultiExtractor):
        super().__init__()
        self._onset_multi_extractor = onset_multi_extractor

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.AUDIO_PATH,
            AudioDataFeature.SAMPLE_RATE_HZ,
            AudioDataFeature.PERCUSSIVE_AUDIO,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            percussive = data[AudioDataFeature.PERCUSSIVE_AUDIO]
            common_args = {
                "file_path": data[AudioDataFeature.AUDIO_PATH],
                "sample_rate": data[AudioDataFeature.SAMPLE_RATE_HZ],
                "audio": percussive,
                "start_sample": 0,
                "end_sample": len(percussive),
                "hop_length": 512,
            }

        onset_envelope_results = self._onset_multi_extractor.extract(**common_args)
        self._add_timed_cache_times(onset_envelope_results)

        with self._measure_processing():
            return {
                AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES: onset_envelope_results.value
            }


class MultiBandOnsetPeaksProvider(AudioDataFeatureProvider):
    """
    Intermediate calculator for multi-band onset peaks.
    Accepts a shared OnsetStrengthMultiExtractor instance to avoid redundant computation.
    """
    def __init__(self, onset_multi_extractor: OnsetStrengthMultiExtractor):
        super().__init__()
        self._onset_multi_extractor = onset_multi_extractor

    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.AUDIO_PATH,
            AudioDataFeature.SAMPLE_RATE_HZ,
            AudioDataFeature.PERCUSSIVE_AUDIO,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.MULTI_BAND_ONSET_PEAKS

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            percussive = data[AudioDataFeature.PERCUSSIVE_AUDIO]
            common_args = {
                "file_path": data[AudioDataFeature.AUDIO_PATH],
                "sample_rate": data[AudioDataFeature.SAMPLE_RATE_HZ],
                "audio": percussive,
                "start_sample": 0,
                "end_sample": len(percussive),
                "hop_length": 512,
            }

        results = self._onset_multi_extractor.extract_peaks(**common_args)
        self._add_timed_cache_times(results)

        with self._measure_processing():
            return {
                AudioDataFeature.MULTI_BAND_ONSET_PEAKS: results.value
            }

# --- KICK Calculators ---

class OnsetEnvMeanKickProvider(AudioDataFeatureProvider):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_ENV_MEAN_KICK

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            onset_envs = data[AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES]
            return {AudioDataFeature.ONSET_ENV_MEAN_KICK: float(onset_envs.get("kick", np.array([0.0])).mean())}


class OnsetRateKickProvider(AudioDataFeatureProvider):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.MULTI_BAND_ONSET_PEAKS,
            AudioDataFeature.AUDIO_SAMPLES,
            AudioDataFeature.SAMPLE_RATE_HZ,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_RATE_KICK

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            onset_peaks = data[AudioDataFeature.MULTI_BAND_ONSET_PEAKS]
            samples = data[AudioDataFeature.AUDIO_SAMPLES]
            sr = data[AudioDataFeature.SAMPLE_RATE_HZ]
            duration_sec = len(samples) / sr if sr > 0 else 1.0
            return {AudioDataFeature.ONSET_RATE_KICK: len(onset_peaks.get("kick", [])) / duration_sec}


# --- SNARE Calculators ---

class OnsetEnvMeanSnareProvider(AudioDataFeatureProvider):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_ENV_MEAN_SNARE

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            onset_envs = data[AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES]
            return {AudioDataFeature.ONSET_ENV_MEAN_SNARE: float(onset_envs.get("snare", np.array([0.0])).mean())}


class OnsetRateSnareProvider(AudioDataFeatureProvider):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.MULTI_BAND_ONSET_PEAKS,
            AudioDataFeature.AUDIO_SAMPLES,
            AudioDataFeature.SAMPLE_RATE_HZ,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_RATE_SNARE

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            onset_peaks = data[AudioDataFeature.MULTI_BAND_ONSET_PEAKS]
            samples = data[AudioDataFeature.AUDIO_SAMPLES]
            sr = data[AudioDataFeature.SAMPLE_RATE_HZ]
            duration_sec = len(samples) / sr if sr > 0 else 1.0
            return {AudioDataFeature.ONSET_RATE_SNARE: len(onset_peaks.get("snare", [])) / duration_sec}


# --- LOW-MID Calculators ---

class OnsetEnvMeanLowMidProvider(AudioDataFeatureProvider):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_ENV_MEAN_LOW_MID

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            onset_envs = data[AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES]
            return {AudioDataFeature.ONSET_ENV_MEAN_LOW_MID: float(onset_envs.get("low_mid", np.array([0.0])).mean())}


class OnsetRateLowMidProvider(AudioDataFeatureProvider):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.MULTI_BAND_ONSET_PEAKS,
            AudioDataFeature.AUDIO_SAMPLES,
            AudioDataFeature.SAMPLE_RATE_HZ,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_RATE_LOW_MID

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            onset_peaks = data[AudioDataFeature.MULTI_BAND_ONSET_PEAKS]
            samples = data[AudioDataFeature.AUDIO_SAMPLES]
            sr = data[AudioDataFeature.SAMPLE_RATE_HZ]
            duration_sec = len(samples) / sr if sr > 0 else 1.0
            return {AudioDataFeature.ONSET_RATE_LOW_MID: len(onset_peaks.get("low_mid", [])) / duration_sec}


# --- HI-HAT Calculators ---

class OnsetEnvMeanHiHatProvider(AudioDataFeatureProvider):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_ENV_MEAN_HI_HAT

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            onset_envs = data[AudioDataFeature.MULTI_BAND_ONSET_ENVELOPES]
            return {AudioDataFeature.ONSET_ENV_MEAN_HI_HAT: float(onset_envs.get("hihat", np.array([0.0])).mean())}


class OnsetRateHiHatProvider(AudioDataFeatureProvider):
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.MULTI_BAND_ONSET_PEAKS,
            AudioDataFeature.AUDIO_SAMPLES,
            AudioDataFeature.SAMPLE_RATE_HZ,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.ONSET_RATE_HI_HAT

    def _provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        with self._measure_processing():
            onset_peaks = data[AudioDataFeature.MULTI_BAND_ONSET_PEAKS]
            samples = data[AudioDataFeature.AUDIO_SAMPLES]
            sr = data[AudioDataFeature.SAMPLE_RATE_HZ]
            duration_sec = len(samples) / sr if sr > 0 else 1.0
            return {AudioDataFeature.ONSET_RATE_HI_HAT: len(onset_peaks.get("hihat", [])) / duration_sec}
