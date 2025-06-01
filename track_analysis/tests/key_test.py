from pathlib import Path
from typing import Optional

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.testing import TestInterface
from track_analysis.components.track_analysis.features.key_extraction.core.config.key_progression_config import \
    KeyProgressionConfig
from track_analysis.components.track_analysis.features.key_extraction.core.definitions.definition_templates import \
    TemplateMode
from track_analysis.components.track_analysis.features.key_extraction.key_progression_analyzer import \
    KeyProgressionAnalyzer


class KeyProgressionTest(TestInterface):
    def __init__(self,
                 logger: HoornLogger,
                 template_mode: TemplateMode = TemplateMode.KS_T_REVISED,
                 tone_modulation_penalty: float = 6.0,
                 mode_modulation_penalty: Optional[float] = None,
                 segment_beat_level: int = 3,
                 visualize: bool = False):
        super().__init__(logger, is_child=True)
        self._separator = 'KeyProgressionTest'
        config: KeyProgressionConfig = KeyProgressionConfig(
            tone_modulation_penalty=tone_modulation_penalty,
            mode_modulation_penalty=mode_modulation_penalty,
            visualize=visualize,
            template_mode=template_mode,
            hop_length=512,
            subdivisions_per_beat=2,
            segment_beat_level=segment_beat_level
        )
        self._analyzer = KeyProgressionAnalyzer(logger, config)

    def test(self, file_path: Path) -> None:
        runs, global_key = self._analyzer.analyze(file_path)
        for run in runs:
            duration = run.end_time - run.start_time
            self._logger.info(
                f"[segment {run.start_index}] {self._format_time(run.start_time)} -> {self._format_time(run.end_time)}"
                f" ({duration:.2f}s) => {run.state_label}",
                separator=self._separator
            )
        self._logger.info(f"The global key is {global_key}", separator=self._separator)
        self._logger.info("Key progression complete.", separator=self._separator)

    @staticmethod
    def _format_time(seconds: float) -> str:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}:{secs:05.2f}"
