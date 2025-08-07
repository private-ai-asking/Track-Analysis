from typing import List, Dict, Any

# Assuming your existing imports are here
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature_provider import \
    AudioDataFeatureProvider

# Assuming BANDS is defined elsewhere, e.g.:
BANDS = {
    "sub_bass": (20, 60),
    "bass": (60, 250),
    "low_mid": (250, 500),
    "mid": (500, 2000),
    "high": (2000, 20000),
}


class MultiBandEnergyProvider(AudioDataFeatureProvider):
    """
    Calculates the total energy within different frequency bands.
    This version depends on pre-computed spectrograms and FFT frequencies.
    """
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [
            AudioDataFeature.HARMONIC_MAGNITUDE_SPECTROGRAM,
            AudioDataFeature.FFT_FREQUENCIES,
        ]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.MULTI_BAND_ENERGY

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        magnitude_spectrogram = data[AudioDataFeature.HARMONIC_MAGNITUDE_SPECTROGRAM]

        freqs = data[AudioDataFeature.FFT_FREQUENCIES]

        power_spec = magnitude_spectrogram**2

        band_energies = {}
        for band_name, (low_freq, high_freq) in BANDS.items():
            band_mask = (freqs >= low_freq) & (freqs < high_freq)
            band_energies[band_name] = power_spec[band_mask, :].sum()

        return {AudioDataFeature.MULTI_BAND_ENERGY: band_energies}

# --- Bass Energy Ratio Provider ---

class BassEnergyRatioProvider(AudioDataFeatureProvider):
    """
    Calculates the ratio of combined bass energy (sub-bass + bass) to total energy.
    This feature helps identify tracks with a dominant low-end presence.
    """
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.MULTI_BAND_ENERGY]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.BASS_ENERGY_RATIO

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        band_energies = data[AudioDataFeature.MULTI_BAND_ENERGY]

        # Sum the energy from the sub-bass and bass bands
        bass_total_energy = band_energies.get("sub_bass", 0.0) + band_energies.get("bass", 0.0)

        # Sum the energy from all bands to get the total
        total_energy = sum(band_energies.values())

        # Calculate the ratio, handling the case of zero total energy
        ratio = bass_total_energy / total_energy if total_energy > 0 else 0.0

        return {AudioDataFeature.BASS_ENERGY_RATIO: ratio}

# --- Sub-Bass Energy Ratio Provider ---

class SubBassEnergyRatioProvider(AudioDataFeatureProvider):
    """
    Calculates the ratio of sub-bass energy (e.g., 20-60Hz) to total energy.
    This feature is specific to the deep "rumble" found in cinematic scores or certain electronic genres.
    """
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.MULTI_BAND_ENERGY]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.SUB_BASS_ENERGY_RATIO

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        band_energies = data[AudioDataFeature.MULTI_BAND_ENERGY]

        sub_bass_energy = band_energies.get("sub_bass", 0.0)
        total_energy = sum(band_energies.values())

        ratio = sub_bass_energy / total_energy if total_energy > 0 else 0.0

        return {AudioDataFeature.SUB_BASS_ENERGY_RATIO: ratio}

# --- Low-Mid Energy Ratio Provider ---

class LowMidEnergyRatioProvider(AudioDataFeatureProvider):
    """
    Calculates the ratio of low-mid energy (e.g., 250-500Hz) to total energy.
    This feature captures the "body" or "mud" region of a mix, which can be important
    for distinguishing between different instrumentations (e.g., heavy guitars vs. clean synths).
    """
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.MULTI_BAND_ENERGY]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.LOW_MID_ENERGY_RATIO

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        band_energies = data[AudioDataFeature.MULTI_BAND_ENERGY]

        low_mid_energy = band_energies.get("low_mid", 0.0)
        total_energy = sum(band_energies.values())

        ratio = low_mid_energy / total_energy if total_energy > 0 else 0.0

        return {AudioDataFeature.LOW_MID_ENERGY_RATIO: ratio}

# --- Mid Energy Ratio Provider ---

class MidEnergyRatioProvider(AudioDataFeatureProvider):
    """
    Calculates the ratio of mid-range energy (e.g., 500-2000Hz) to total energy.
    This feature captures the core melodic and vocal presence of a track.
    """
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.MULTI_BAND_ENERGY]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.MID_ENERGY_RATIO

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        band_energies = data[AudioDataFeature.MULTI_BAND_ENERGY]

        mid_energy = band_energies.get("mid", 0.0)
        total_energy = sum(band_energies.values())

        ratio = mid_energy / total_energy if total_energy > 0 else 0.0

        return {AudioDataFeature.MID_ENERGY_RATIO: ratio}

# --- High Energy Ratio Provider ---

class HighEnergyRatioProvider(AudioDataFeatureProvider):
    """
    Calculates the ratio of high-frequency energy (e.g., 2000-20000Hz) to total energy.
    This feature captures the "brightness", "air", and "crispness" of a track.
    """
    @property
    def dependencies(self) -> List[AudioDataFeature]:
        return [AudioDataFeature.MULTI_BAND_ENERGY]

    @property
    def output_features(self) -> AudioDataFeature:
        return AudioDataFeature.HIGH_ENERGY_RATIO

    def provide(self, data: Dict[AudioDataFeature, Any]) -> Dict[AudioDataFeature, Any]:
        band_energies = data[AudioDataFeature.MULTI_BAND_ENERGY]

        high_energy = band_energies.get("high", 0.0)
        total_energy = sum(band_energies.values())

        ratio = high_energy / total_energy if total_energy > 0 else 0.0

        return {AudioDataFeature.HIGH_ENERGY_RATIO: ratio}
