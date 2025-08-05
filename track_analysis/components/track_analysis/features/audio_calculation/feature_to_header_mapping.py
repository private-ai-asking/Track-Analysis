from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.data_generation.model.header import Header

# This dictionary explicitly maps all the features we want to calculate
# to the final column names in the output DataFrames.
FEATURE_TO_HEADER_MAPPING = {
    # BASE FEATURES
    AudioDataFeature.DATA_RATE: Header.Actual_Data_Rate,
    AudioDataFeature.MAX_DATA_RATE: Header.Max_Data_Per_Second,
    AudioDataFeature.DATA_EFFICIENCY: Header.Efficiency,
    AudioDataFeature.DURATION: Header.Duration,
    AudioDataFeature.SAMPLE_RATE_KHZ: Header.Sample_Rate,
    AudioDataFeature.BIT_DEPTH: Header.Bit_Depth,
    AudioDataFeature.BIT_RATE: Header.Bitrate,
    AudioDataFeature.BPM: Header.BPM,

    # Loudness Features
    AudioDataFeature.INTEGRATED_LUFS: Header.Integrated_LUFS,
    AudioDataFeature.LOUDNESS_RANGE_LU: Header.Program_Dynamic_Range_LRA,
    AudioDataFeature.TRUE_PEAK: Header.True_Peak,
    AudioDataFeature.CREST_FACTOR: Header.Crest_Factor,

    # RMS Features
    AudioDataFeature.RMS_MEAN: Header.Mean_RMS,
    AudioDataFeature.RMS_MAX: Header.Max_RMS,
    AudioDataFeature.RMS_PERC: Header.Percentile_90_RMS,
    AudioDataFeature.RMS_IQR: Header.RMS_IQR,

    # Harmonicity & Tempo
    AudioDataFeature.HARMONICITY: Header.Harmonicity,
    AudioDataFeature.TEMPO_VARIATION: Header.Tempo_Variation,

    # Spectral Features
    AudioDataFeature.SPECTRAL_CENTROID_MEAN: Header.Spectral_Centroid_Mean,
    AudioDataFeature.SPECTRAL_CENTROID_MAX: Header.Spectral_Centroid_Max,
    AudioDataFeature.SPECTRAL_CENTROID_STD: Header.Spectral_Centroid_Std,
    AudioDataFeature.SPECTRAL_FLUX_MEAN: Header.Spectral_Flux_Mean,
    AudioDataFeature.SPECTRAL_FLUX_MAX: Header.Spectral_Flux_Max,
    AudioDataFeature.SPECTRAL_FLUX_STD: Header.Spectral_Flux_Std,
    AudioDataFeature.SPECTRAL_ROLLOFF_MEAN: Header.Spectral_Rolloff_Mean,
    AudioDataFeature.SPECTRAL_ROLLOFF_STD: Header.Spectral_Rolloff_Std,
    AudioDataFeature.SPECTRAL_FLATNESS_MEAN: Header.Spectral_Flatness_Mean,
    AudioDataFeature.SPECTRAL_CONTRAST_MEAN: Header.Spectral_Contrast_Mean,
    AudioDataFeature.ZCR_MEAN: Header.Zero_Crossing_Rate_Mean,

    # Onset Features
    AudioDataFeature.ONSET_ENV_MEAN_GLOBAL: Header.Onset_Env_Mean,
    AudioDataFeature.ONSET_RATE_GLOBAL: Header.Onset_Rate,
    AudioDataFeature.ONSET_ENV_MEAN_KICK: Header.Onset_Env_Mean_Kick,
    AudioDataFeature.ONSET_RATE_KICK: Header.Onset_Rate_Kick,
    AudioDataFeature.ONSET_ENV_MEAN_SNARE: Header.Onset_Env_Mean_Snare,
    AudioDataFeature.ONSET_RATE_SNARE: Header.Onset_Rate_Snare,
    AudioDataFeature.ONSET_ENV_MEAN_LOW_MID: Header.Onset_Env_Mean_Low_Mid,
    AudioDataFeature.ONSET_RATE_LOW_MID: Header.Onset_Rate_Low_Mid,
    AudioDataFeature.ONSET_ENV_MEAN_HI_HAT: Header.Onset_Env_Mean_Hi_Hat,
    AudioDataFeature.ONSET_RATE_HI_HAT: Header.Onset_Rate_Hi_Hat,
}

HEADER_TO_FEATURE_MAPPING = {v: k for k, v in FEATURE_TO_HEADER_MAPPING.items()}
