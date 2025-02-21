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

    BPM = "BPM"
    Energy_Level = "Energy Level"
    Key = "Key"

    Duration = "Duration"
    Bitrate = "Bitrate"
    Sample_Rate = "Sample Rate"
    Dynamic_Range = "Dynamic Range"
