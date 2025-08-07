from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.core.config.key_progression_config import \
    KeyProgressionConfig
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.core.definitions.definition_templates import \
    TemplateMode

DEFAULT_KEY_PROGRESSION_CONFIG = KeyProgressionConfig(
    tone_modulation_penalty=18,
    mode_modulation_penalty=None,
    visualize=False,
    template_mode=TemplateMode.HOORN,
    hop_length=512,
    subdivisions_per_beat=2,
    min_segment_beat_level=4
)
