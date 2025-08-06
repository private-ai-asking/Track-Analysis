from pathlib import Path
from typing import List, Tuple, Optional, Dict

from track_analysis.components.md_common_python.py_common.algorithms.sequence.run_length_merger import StateRun
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.legacy.audio_file_handler import AudioFileHandler
from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.providers.calculated.rhythm.tempo_provider import \
    BeatDetector
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.core.config.key_progression_config import \
    KeyProgressionConfig
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.extractors.global_key_extractor import \
    GlobalKeyEstimator
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.extractors.local_key_extractor import \
    LocalKeyEstimator
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.preprocessing.note_extraction.notes.note_event_builder import \
    NoteEvent

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


class KeyProgressionAnalyzer:
    """
    Thin orchestrator. Its only job is:
      1) Load audio,
      2) Detect beats,
      3) Hand off to LocalKeyEstimator,
      4) Hand off to GlobalKeyEstimator,
      5) Return (local_runs, global_key).
    """
    def __init__(self, logger: HoornLogger, config: KeyProgressionConfig, audio_loader: AudioFileHandler) -> None:
        self._logger = logger
        self._separator = self.__class__.__name__
        self._config = config

        # 1. Subcomponents that KeyProgressionAnalyzer itself must own:
        self._audio_loader = audio_loader
        self._beat_detector = BeatDetector(logger)

        # 2. Local‐ vs. Global‐key estimators:
        self._local_estimator = LocalKeyEstimator(logger, config)
        local_templates = self._local_estimator.get_local_templates()

        self._global_estimator = GlobalKeyEstimator(logger, local_templates)

        self._logger.info("Initialized KeyProgressionAnalyzer.", separator=self._separator)

    def analyze(self, file_path: Path) -> Tuple[Optional[List[StateRun]], Optional[str], Optional[List[NoteEvent]]]:
        """
        Main entry point. Returns:
          - List of StateRun (local key progression)
          - global key (str), or None if file is missing/not readable
          - List of NoteEvents for the track
        """
        if not file_path.is_file():
            self._logger.error(f"File not found: {file_path}", separator=self._separator)
            return [], None, None

        self._logger.info(f"Starting analysis on file: {file_path}", separator=self._separator)

        # 1) Load audio from disk:
        info = self._audio_loader.get_audio_streams_info_batch([file_path])[0]
        audio_samples = info.samples
        sample_rate = info.sample_rate_Hz

        # 2) Detect beats (tempo + frame indices + times):
        tempo = self._beat_detector.get_tempo(audio_path=file_path, audio=audio_samples, sample_rate=sample_rate, onset_envelope=None)
        beat_frames, beat_times = self._beat_detector.get_beat_frames_and_times(audio_path=file_path, audio=audio_samples, sample_rate=sample_rate, onset_envelope=None)

        if tempo == 0:
            self._logger.warning(f"[Tempo = 0] Local key estimation failed for: \"{file_path}\"...", separator=self._separator)
            return None, None, None

        # 3) Ask LocalKeyEstimator to do everything up through local runs:
        result = self._local_estimator.analyze(
            info.path, audio_samples, sample_rate, tempo, beat_frames, beat_times
        )

        if result is None:
            self._logger.warning(f"[Local analysis failed] Local key estimation failed for: \"{file_path}\"...", separator=self._separator)
            return None, None, None

        local_runs = result[0]
        feature_matrix = result[2]
        notes = result[3]

        # 4) Ask GlobalKeyEstimator to pick one global key:
        global_key = self._global_estimator.estimate_global_key(
            feature_matrix
        )

        self._logger.info("Key progression analysis complete.", separator=self._separator)
        return local_runs, global_key, notes

    def convert_label_to_camelot(self, label: str) -> str:
        camelot: str = _CAMELOT_MAPPING.get(label, None)

        if camelot is None:
            self._logger.warning(f"No camelot mapping found for {label}, returning label.", separator=self._separator)
            return label

        return camelot
