from dataclasses import dataclass


@dataclass(frozen=True)
class StateLabelInfo:
    index: int
    tonic: str
    mode: str
