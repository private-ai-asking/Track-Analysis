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
    Spectral_Centroid_Mean = "Spectral Centroid Mean (Hz)"
    Spectral_Centroid_Max  = "Spectral Centroid Max (Hz)"
    Spectral_Flux_Mean     = "Spectral Flux Mean"
    Spectral_Flux_Max      = "Spectral Flux Max"
    Onset_Env_Mean         = "Mean Onset Strength"
    Onset_Rate             = "Onset Rate (events/sec)"
    Onset_Rate_Notes = "Onset Rate (note events/sec)"

