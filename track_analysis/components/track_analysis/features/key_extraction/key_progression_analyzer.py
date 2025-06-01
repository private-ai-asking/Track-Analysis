from pathlib import Path
from typing import List, Tuple, Optional

from track_analysis.components.md_common_python.py_common.algorithms.sequence.run_length_merger import StateRun
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.key_extraction.core.config.key_progression_config import \
    KeyProgressionConfig
from track_analysis.components.track_analysis.features.key_extraction.core.extraction.global_key_extractor import \
    GlobalKeyEstimator
from track_analysis.components.track_analysis.features.key_extraction.core.extraction.local_key_extractor import \
    LocalKeyEstimator
from track_analysis.components.track_analysis.features.key_extraction.utils.audio_loader import AudioLoader
from track_analysis.components.track_analysis.features.key_extraction.utils.beat_detector import BeatDetector


class KeyProgressionAnalyzer:
    """
    Thin orchestrator. Its only job is:
      1) Load audio,
      2) Detect beats,
      3) Hand off to LocalKeyEstimator,
      4) Hand off to GlobalKeyEstimator,
      5) Return (local_runs, global_key).
    """
    def __init__(self, logger: HoornLogger, config: KeyProgressionConfig) -> None:
        self._logger = logger
        self._separator = self.__class__.__name__
        self._config = config

        # 1. Subcomponents that KeyProgressionAnalyzer itself must own:
        self._audio_loader = AudioLoader(logger, config.cache_dir / "audio_loading")
        self._beat_detector = BeatDetector(logger)

        # 2. Local‐ vs. Global‐key estimators:
        self._local_estimator = LocalKeyEstimator(logger, config)
        local_templates = self._local_estimator.get_local_templates()

        self._global_estimator = GlobalKeyEstimator(logger, local_templates)

        self._logger.info("Initialized KeyProgressionAnalyzer.", separator=self._separator)

    def analyze(self, file_path: Path) -> Tuple[List[StateRun], Optional[str]]:
        """
        Main entry point. Returns:
          - List of StateRun (local key progression)
          - global key (str), or None if file is missing/not readable
        """
        if not file_path.is_file():
            self._logger.error(f"File not found: {file_path}", separator=self._separator)
            return [], None

        self._logger.info(f"Starting analysis on file: {file_path}", separator=self._separator)

        # 1) Load audio from disk:
        audio_samples, sample_rate = self._audio_loader.load(file_path)

        # 2) Detect beats (tempo + frame indices + times):
        tempo, beat_frames, beat_times = self._beat_detector.detect(audio_samples, sample_rate)

        # 3) Ask LocalKeyEstimator to do everything up through local runs:
        local_runs, intervals, feature_matrix = self._local_estimator.analyze(
            audio_samples, sample_rate, tempo, beat_frames, beat_times
        )

        # 4) Ask GlobalKeyEstimator to pick one global key:
        global_key = self._global_estimator.estimate_global_key(
            feature_matrix, intervals
        )

        self._logger.info("Key progression analysis complete.", separator=self._separator)
        return local_runs, global_key
