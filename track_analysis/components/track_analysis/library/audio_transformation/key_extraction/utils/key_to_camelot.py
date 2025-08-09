from typing import Dict

_CAMELOT_MAPPING: Dict[str, str] = {
    # 1A: Ab minor / G# minor
    "Ab Aeolian (Minor)": "1A",
    "G# Aeolian (Minor)": "1A",

    # 1B: B major / Cb major
    "B Ionian (Major)": "1B",
    "Cb Ionian (Major)": "1B",

    # 2A: Eb minor / D# minor
    "Eb Aeolian (Minor)": "2A",
    "D# Aeolian (Minor)": "2A",

    # 2B: F# major / Gb major
    "F# Ionian (Major)": "2B",
    "Gb Ionian (Major)": "2B",

    # 3A: Bb minor / A# minor
    "Bb Aeolian (Minor)": "3A",
    "A# Aeolian (Minor)": "3A",

    # 3B: Db major / C# major
    "Db Ionian (Major)": "3B",
    "C# Ionian (Major)": "3B",

    # 4A: F minor
    "F Aeolian (Minor)": "4A",

    # 4B: Ab major / G# major
    "Ab Ionian (Major)": "4B",
    "G# Ionian (Major)": "4B",

    # 5A: C minor
    "C Aeolian (Minor)": "5A",

    # 5B: Eb major / D# major
    "Eb Ionian (Major)": "5B",
    "D# Ionian (Major)": "5B",

    # 6A: G minor
    "G Aeolian (Minor)": "6A",

    # 6B: Bb major / A# major
    "Bb Ionian (Major)": "6B",
    "A# Ionian (Major)": "6B",

    # 7A: D minor
    "D Aeolian (Minor)": "7A",

    # 7B: F major
    "F Ionian (Major)": "7B",

    # 8A: A minor
    "A Aeolian (Minor)": "8A",

    # 8B: C major
    "C Ionian (Major)": "8B",

    # 9A: E minor
    "E Aeolian (Minor)": "9A",

    # 9B: G major
    "G Ionian (Major)": "9B",

    # 10A: B minor
    "B Aeolian (Minor)": "10A",

    # 10B: D major
    "D Ionian (Major)": "10B",

    # 11A: F# minor / Gb minor
    "F# Aeolian (Minor)": "11A",
    "Gb Aeolian (Minor)": "11A",

    # 11B: A major
    "A Ionian (Major)": "11B",

    # 12A: Db minor / C# minor
    "Db Aeolian (Minor)": "12A",
    "C# Aeolian (Minor)": "12A",

    # 12B: E major / Fb major
    "E Ionian (Major)": "12B",
    "Fb Ionian (Major)": "12B",
}

def convert_label_to_camelot(label: str) -> str:
    camelot: str = _CAMELOT_MAPPING.get(label, None)

    if camelot is None:
        raise ValueError(f"{label} is not a valid camelot label")

    return camelot
