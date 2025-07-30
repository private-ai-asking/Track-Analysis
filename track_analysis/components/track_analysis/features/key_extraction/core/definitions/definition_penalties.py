from typing import Dict

MODE_PENALTIES: Dict[str, Dict[str, float]] = {
    "Ionian (Major)": {
        "Ionian (Major)": 0.0,
        "Aeolian (Minor)": 3.0,
        "Dorian (Minor)": 2.0,
    },
    "Aeolian (Minor)": {
        "Aeolian (Minor)": 0.0,
        "Ionian (Major)": 3.0,
        "Dorian (Minor)": 1.0,
    },
    "Dorian (Minor)": {
        "Dorian (Minor)": 0.0,
        "Ionian (Major)": 2.0,
        "Aeolian (Minor)": 1.0,
    }
}
