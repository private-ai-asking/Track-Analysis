from dataclasses import dataclass

@dataclass(frozen=True)
class NoteEvent:
    pitch_class: int
    onset_seconds: float
    offset_seconds: float
