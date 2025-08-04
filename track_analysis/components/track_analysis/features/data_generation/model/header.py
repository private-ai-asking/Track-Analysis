from enum import Enum


class Header(Enum):
    UUID = "UUID"

    Title = "Title"
    Album = "Album"
    Artists = "Artist(s)"
    Primary_Artist = "Primary Artist"
    Album_Artists = "Album Artist(s)"
    Label = "Label"
    Extension = "Audio File Extension"
    Audio_Path = "Audio Path"

    Release_Year = "Release Year"
    Release_Date = "Release Date"

    Genre = "Genre"
    Bought = "Bought"
    Album_Cost = "Album Cost"

    BPM = "BPM"
    Energy_Level = "Energy Level"

    Key = "Principal Key"
    Start_Key = "Start Key"
    End_Key = "End Key"

    Duration = "Duration"
    Bitrate = "Bitrate"
    Sample_Rate = "Sample Rate"
    Program_Dynamic_Range_LRA = "Program Dynamic Range (LRA proxy -> LU)"
    Crest_Factor = "Crest Factor (peak-to-RMS/dB)"
    Bit_Depth = "Bit Depth"
    Max_Data_Per_Second = "Max Data Per Second (Kilobits)"
    Actual_Data_Rate = "Actual Data Rate (Kilobits)"
    Efficiency = "Data Efficiency"
    Format = "Audio Format"
    Integrated_LUFS = "Integrated LUFS"
    True_Peak = "True Peak (dBPT)"
    Mean_RMS = "Mean RMS (dBFS)"
    Max_RMS = "Max RMS (dBFS)"
    Percentile_90_RMS = "90th Percentile RMS (dBFS)"
    RMS_IQR = "RMS Dynamic Range (IQR dBFS)"
    Tempo_Variation = "Tempo Variation (std)"
    Harmonicity = "Harmonicity (0-1)"

    # Spectral
    Spectral_Centroid_Mean = "Spectral Centroid Mean (Hz)"
    Spectral_Centroid_Max  = "Spectral Centroid Max (Hz)"
    Spectral_Centroid_Std = "Spectral Centroid Std (Hz)"
    Spectral_Flux_Mean     = "Spectral Flux Mean"
    Spectral_Flux_Max      = "Spectral Flux Max"
    Spectral_Flux_Std = "Spectral Flux Std"
    Spectral_Rolloff_Mean = "Spectral Rolloff (mean)"
    Spectral_Rolloff_Std = "Spectral Rolloff (std)"
    Zero_Crossing_Rate_Mean = "Zero Crossing Rate Mean"
    Spectral_Flatness_Mean = "Spectral Flatness Mean"
    Spectral_Contrast_Mean = "Spectral Contrast Mean"

    # Onsets
    Onset_Env_Mean         = "Mean Onset Strength [Global]"
    Onset_Rate             = "Onset Rate (events/sec) [Global]"

    Onset_Env_Mean_Kick = "Mean Onset Strength [Kick]"
    Onset_Rate_Kick = "Onset Rate (events/sec) [Kick]"

    Onset_Env_Mean_Snare = "Mean Onset Strength [Snare]"
    Onset_Rate_Snare = "Onset Rate (events/sec) [Snare]"

    Onset_Env_Mean_Low_Mid = "Mean Onset Strength [Low Mid]"
    Onset_Rate_Low_Mid = "Onset Rate (events/sec) [Low Mid]"

    Onset_Env_Mean_Hi_Hat = "Mean Onset Strength [Hi Hat]"
    Onset_Rate_Hi_Hat = "Onset Rate (events/sec) [Hi Hat]"

    # SPECIAL
    MFCC = "Not used... Special for CSV flow"

