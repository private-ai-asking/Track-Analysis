from enum import Enum


class Header(Enum):
    """
    Enumeration for header columns, organized logically for display and processing.
    The order flows from general track metadata to technical file properties,
    and finally to detailed, calculated audio features.
    """

    # //--- Core Metadata & Identifiers ---//
    # Standard tags for identifying and categorizing the track.
    UUID = "UUID"
    Title = "Title"
    Artists = "Artist(s)"
    Primary_Artist = "Primary Artist"
    Album = "Album"
    Album_Artists = "Album Artist(s)"
    Label = "Label"
    Genre = "Genre"
    Release_Year = "Release Year"
    Release_Date = "Release Date"
    Bought = "Bought"
    Album_Cost = "Album Cost"

    # //--- File & Technical Properties ---//
    # Intrinsic properties of the audio file itself.
    Audio_Path = "Audio Path"
    Extension = "Audio File Extension"
    Duration = "Duration"
    Sample_Rate = "Sample Rate"
    Bit_Depth = "Bit Depth"
    Bitrate = "Bitrate"
    Max_Data_Per_Second = "Max Data Per Second (Kilobits)"
    Actual_Data_Rate = "Actual Data Rate (Kilobits)"
    Efficiency = "Data Efficiency"

    # //--- High-Level Musical Descriptors ---//
    # Simple, top-level musical characteristics.
    BPM = "BPM"
    Energy_Level = "Energy Level"
    Key = "Principal Key"
    Start_Key = "Start Key"
    End_Key = "End Key"

    # //--- Loudness & Dynamic Range Features ---//
    # Features related to perceived volume and dynamics, following EBU R 128 standards.
    Integrated_LUFS = "Integrated LUFS"
    Integrated_LUFS_Mean = "Integrated LUFS (mean)"
    Integrated_LUFS_STD = "Integrated LUFS Variation (std)"
    Integrated_LUFS_Range = "Integrated LUFS (range)"
    Program_Dynamic_Range_LRA = "Program Dynamic Range (LRA proxy -> LU)"
    True_Peak = "True Peak (dBPT)"
    Crest_Factor = "Crest Factor (peak-to-RMS/dB)"
    Mean_RMS = "Mean RMS (dBFS)"
    Max_RMS = "Max RMS (dBFS)"
    Percentile_90_RMS = "90th Percentile RMS (dBFS)"
    RMS_IQR = "RMS Dynamic Range (IQR dBFS)"

    # //--- Rhythmic & Tonal Features ---//
    # Features describing rhythm, harmony, and percussive content.
    Tempo_Variation = "Tempo Variation (std)"
    Beat_Strength = "Beat Strength"
    Rhythmic_Regularity = "Rhythmic Regularity"
    Harmonicity = "Harmonicity (0-1)"
    HPR = "Harmonic to Percussive Ratio"
    Chroma_Entropy = "Chroma Entropy"

    # //--- Spectral Features ---//
    # Advanced features describing the timbre and frequency content of the audio.
    Spectral_Centroid_Mean = "Spectral Centroid Mean (Hz)"
    Spectral_Centroid_Std = "Spectral Centroid Std (Hz)"
    Spectral_Centroid_Max = "Spectral Centroid Max (Hz)"
    Spectral_Flux_Mean = "Spectral Flux Mean"
    Spectral_Flux_Std = "Spectral Flux Std"
    Spectral_Flux_Max = "Spectral Flux Max"
    Spectral_Rolloff_Mean = "Spectral Rolloff (mean)"
    Spectral_Rolloff_Std = "Spectral Rolloff (std)"
    Spectral_Flatness_Mean = "Spectral Flatness Mean"
    Spectral_Flatness_STD = "Spectral Flatness STD"
    Spectral_Contrast_Mean = "Spectral Contrast Mean"
    Spectral_Contrast_STD = "Spectral Contrast STD"
    Spectral_Skewness = "Spectral Skewness"
    Spectral_Kurtosis = "Spectral Kurtosis"
    Spectral_Entropy = "Spectral Entropy"
    Zero_Crossing_Rate_Mean = "Zero Crossing Rate Mean"
    Zero_Crossing_Rate_STD = "Zero Crossing Rate STD"
    Spectral_Bandwidth_Mean = "Spectral Bandwidth Mean"
    Spectral_Bandwidth_STD = "Spectral Bandwidth STD"

    # //--- Spectral Band Ratios ---//
    Sub_Bass_Ratio = "Sub-Bass Ratio (20-60hz)"
    Bass_Ratio = "Bass Ratio (60-250hz)"
    Low_Mid_Ratio = "Low Mid Ratio (250-500hz)"
    Mid_Ratio = "Mid Ratio (500-2000hz)"
    High_Ratio = "High Ratio (2000-20000hz)"

    # //--- Onset & Transient Features ---//
    # Features describing the start of musical events (transients) across frequency bands.
    Onset_Env_Mean = "Mean Onset Strength [Global]"
    Onset_Rate = "Onset Rate (events/sec) [Global]"
    Onset_Env_Mean_Kick = "Mean Onset Strength [Kick]"
    Onset_Rate_Kick = "Onset Rate (events/sec) [Kick]"
    Onset_Env_Mean_Snare = "Mean Onset Strength [Snare]"
    Onset_Rate_Snare = "Onset Rate (events/sec) [Snare]"
    Onset_Env_Mean_Low_Mid = "Mean Onset Strength [Low Mid]"
    Onset_Rate_Low_Mid = "Onset Rate (events/sec) [Low Mid]"
    Onset_Env_Mean_Hi_Hat = "Mean Onset Strength [Hi Hat]"
    Onset_Rate_Hi_Hat = "Onset Rate (events/sec) [Hi Hat]"
    Onset_Rate_Variation = "Onset Rate Variation"

    # //--- Internal & Special Use ---//
    # Reserved for internal application logic.
    MFCC = "Not used... Special for CSV flow (mfccs)"
    Key_Progression = "Not used... Special for CSV flow (key)"
