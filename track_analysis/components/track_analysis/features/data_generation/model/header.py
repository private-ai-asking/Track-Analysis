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
    Key = "Key"

    Duration = "Duration"
    Bitrate = "Bitrate"
    Sample_Rate = "Sample Rate"
    Program_Dynamic_Range_LRA = "Program Dynamic Range (LRA proxy)"
    Crest_Factor = "Crest Factor (peak-to-RMS/dB)"
    Bit_Depth = "Bit Depth"
    Max_Data_Per_Second = "Max Data Per Second (Kilobits)"
    Actual_Data_Rate = "Actual Data Rate (Kilobits)"
    Efficiency = "Data Efficiency"
    Format = "Audio Format"
    Integrated_LUFS = "Integrated LUFS"
    True_Peak = "True Peak (dBPT)"

