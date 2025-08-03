from track_analysis.components.track_analysis.features.data_generation.energy_calculation.default.model.energy_model_config import \
    EnergyModelConfig
from track_analysis.components.track_analysis.features.data_generation.model.header import Header

# Defines the standard configuration for the default energy model (version 1).
# This configuration is used to ensure that training and prediction are always
# performed with the same set of well-chosen audio features.

DEFAULT_ENERGY_MODEL_CONFIG = EnergyModelConfig(
    name="default_v1",
    feature_columns=[
        # --- Loudness & Dynamics Features ---
        # These features measure the track's overall volume and compression.
        Header.Integrated_LUFS,          # Overall loudness standard (LUFS).
        Header.Mean_RMS,                 # Average power (loudness).
        Header.Percentile_90_RMS,        # Loudness of the louder sections.
        Header.Program_Dynamic_Range_LRA, # Macro-dynamics; difference between loud and soft parts.
        Header.Crest_Factor,             # Micro-dynamics; ratio of peak to average power.
        Header.RMS_IQR,                  # Dynamic range based on the interquartile range of RMS.

        # --- Rhythmic Features ---
        # These features capture the tempo and the rate of percussive events.
        Header.BPM,                      # Overall tempo in beats per minute.
        Header.Onset_Rate,               # Global rate of all detected note onsets.
        Header.Onset_Rate_Kick,          # Specific rate of kick drum onsets.
        Header.Onset_Rate_Snare,         # Specific rate of snare drum onsets.
        Header.Onset_Rate_Hi_Hat,        # Specific rate of hi-hat onsets.

        # --- Timbre & Texture Features ---
        # These features describe the sonic character, "punchiness," and texture of the track.
        Header.Onset_Env_Mean,           # Average "punchiness" of all onsets.
        Header.Onset_Env_Mean_Kick,      # Average punchiness of kick drums.
        Header.Onset_Env_Mean_Snare,     # Average punchiness of snares.
        Header.Onset_Env_Mean_Low_Mid,   # Punchiness in the low-mid frequency range.
        Header.Onset_Env_Mean_Hi_Hat,    # Punchiness of hi-hats.
        Header.Spectral_Centroid_Mean,   # Average "brightness" or "sharpness" of the sound.
        Header.Spectral_Contrast_Mean,   # Measures the difference between spectral peaks and valleys.
        Header.Zero_Crossing_Rate_Mean,  # Another measure of brightness/noisiness.
        Header.Spectral_Flatness_Mean,   # How "noise-like" versus "tonal" the sound is.
        Header.Spectral_Flux_Mean,       # Rate of change in the timbre over time.
    ]
)
