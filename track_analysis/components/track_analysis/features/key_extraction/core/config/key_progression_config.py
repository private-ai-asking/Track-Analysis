from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from track_analysis.components.track_analysis.constants import CACHE_DIRECTORY
from track_analysis.components.track_analysis.features.key_extraction.core.definitions.definition_templates import \
    TemplateMode


@dataclass(frozen=True)
class KeyProgressionConfig:
    """
    Holds all user‐tunable parameters for key progression analysis.
    """
    tone_modulation_penalty: float = 6.0
    mode_modulation_penalty: Optional[float] = None
    visualize: bool = False
    template_mode: TemplateMode = TemplateMode.KS_T_REVISED

    # Audio/segmentation parameters:
    hop_length: int = 512
    subdivisions_per_beat: int = 2
    segment_beat_level: int = 3

    # Where to cache any intermediate files:
    cache_dir: Path = CACHE_DIRECTORY

    def __post_init__(self):
        if self.mode_modulation_penalty is None:
            object.__setattr__(self, "mode_modulation_penalty", (2.5 / 6) * self.tone_modulation_penalty)
