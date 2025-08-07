from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.data_generation.model.header import Header

# This dictionary explicitly maps all the features we want to calculate
# to the final column names in the output DataFrames. The structure mirrors
# the logical grouping in the Header enum.
FEATURE_TO_HEADER_MAPPING = {
    # //--- File & Technical Properties ---//
    AudioDataFeature.DURATION: Header.Duration,
    AudioDataFeature.SAMPLE_RATE_KHZ: Header.Sample_Rate,
    AudioDataFeature.BIT_DEPTH: Header.Bit_Depth,
    AudioDataFeature.BIT_RATE: Header.Bitrate,
    AudioDataFeature.DATA_RATE: Header.Actual_Data_Rate,
    AudioDataFeature.MAX_DATA_RATE: Header.Max_Data_Per_Second,
    AudioDataFeature.DATA_EFFICIENCY: Header.Efficiency,

    # //--- High-Level Musical Descriptors ---//
    AudioDataFeature.BPM: Header.BPM,
    AudioDataFeature.ENERGY_LEVEL: Header.Energy_Level,

    # //--- Loudness & Dynamic Range Features ---//
    AudioDataFeature.INTEGRATED_LUFS: Header.Integrated_LUFS,
    AudioDataFeature.INTEGRATED_LUFS_MEAN: Header.Integrated_LUFS_Mean,
    AudioDataFeature.INTEGRATED_LUFS_STD: Header.Integrated_LUFS_STD,
    AudioDataFeature.INTEGRATED_LUFS_RANGE: Header.Integrated_LUFS_Range,
    AudioDataFeature.LOUDNESS_RANGE_LU: Header.Program_Dynamic_Range_LRA,
    AudioDataFeature.TRUE_PEAK: Header.True_Peak,
    AudioDataFeature.CREST_FACTOR: Header.Crest_Factor,
    AudioDataFeature.RMS_MEAN: Header.Mean_RMS,
    AudioDataFeature.RMS_MAX: Header.Max_RMS,
    AudioDataFeature.RMS_PERC: Header.Percentile_90_RMS,
    AudioDataFeature.RMS_IQR: Header.RMS_IQR,

    # //--- Rhythmic & Tonal Features ---//
    AudioDataFeature.TEMPO_VARIATION: Header.Tempo_Variation,
    AudioDataFeature.BEAT_STRENGTH: Header.Beat_Strength,
    AudioDataFeature.RHYTHMIC_REGULARITY: Header.Rhythmic_Regularity,
    AudioDataFeature.HARMONICITY: Header.Harmonicity,
    AudioDataFeature.HPR: Header.HPR,
    AudioDataFeature.CHROMA_ENTROPY: Header.Chroma_Entropy,

    # //--- Spectral Features ---//
    AudioDataFeature.SPECTRAL_CENTROID_MEAN: Header.Spectral_Centroid_Mean,
    AudioDataFeature.SPECTRAL_CENTROID_STD: Header.Spectral_Centroid_Std,
    AudioDataFeature.SPECTRAL_CENTROID_MAX: Header.Spectral_Centroid_Max,
    AudioDataFeature.SPECTRAL_FLUX_MEAN: Header.Spectral_Flux_Mean,
    AudioDataFeature.SPECTRAL_FLUX_STD: Header.Spectral_Flux_Std,
    AudioDataFeature.SPECTRAL_FLUX_MAX: Header.Spectral_Flux_Max,
    AudioDataFeature.SPECTRAL_ROLLOFF_MEAN: Header.Spectral_Rolloff_Mean,
    AudioDataFeature.SPECTRAL_ROLLOFF_STD: Header.Spectral_Rolloff_Std,
    AudioDataFeature.SPECTRAL_FLATNESS_MEAN: Header.Spectral_Flatness_Mean,
    AudioDataFeature.SPECTRAL_FLATNESS_STD: Header.Spectral_Flatness_STD,
    AudioDataFeature.SPECTRAL_CONTRAST_MEAN: Header.Spectral_Contrast_Mean,
    AudioDataFeature.SPECTRAL_CONTRAST_STD: Header.Spectral_Contrast_STD,
    AudioDataFeature.SPECTRAL_SKEWNESS: Header.Spectral_Skewness,
    AudioDataFeature.SPECTRAL_KURTOSIS: Header.Spectral_Kurtosis,
    AudioDataFeature.SPECTRAL_ENTROPY: Header.Spectral_Entropy,
    AudioDataFeature.ZCR_MEAN: Header.Zero_Crossing_Rate_Mean,
    AudioDataFeature.ZCR_STD: Header.Zero_Crossing_Rate_STD,
    AudioDataFeature.SPECTRAL_BANDWIDTH_MEAN: Header.Spectral_Bandwidth_Mean,
    AudioDataFeature.SPECTRAL_BANDWIDTH_STD: Header.Spectral_Bandwidth_STD,

    # //--- Spectral Band Ratios ---//
    AudioDataFeature.SUB_BASS_ENERGY_RATIO: Header.Sub_Bass_Ratio,
    AudioDataFeature.BASS_ENERGY_RATIO: Header.Bass_Ratio,
    AudioDataFeature.LOW_MID_ENERGY_RATIO: Header.Low_Mid_Ratio,
    AudioDataFeature.MID_ENERGY_RATIO: Header.Mid_Ratio,
    AudioDataFeature.HIGH_ENERGY_RATIO: Header.High_Ratio,

    # //--- Onset & Transient Features ---//
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
    AudioDataFeature.ONSET_RATE_VARIATION: Header.Onset_Rate_Variation,
}

HEADER_TO_FEATURE_MAPPING = {v: k for k, v in FEATURE_TO_HEADER_MAPPING.items()}
