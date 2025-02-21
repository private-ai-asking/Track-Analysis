from enum import Enum


class Header(Enum):
    Title = "Title"
    Album = "Album"
    Artists = "Artist(s)"
    Album_Artists = "Album Artist(s)"
    Label = "Label"

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
    Peak_To_RMS = "Dynamic Range (peak-to-RMS)"
    Crest_Factor = "Crest Factor"
    Bit_Depth = "Bit Depth"
