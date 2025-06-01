from pathlib import Path
from typing import List

from track_analysis.components.track_analysis.features.key_extraction.core.parsing.parse_label import parse_label
from track_analysis.components.track_analysis.features.key_extraction.feature.vector.feature_vector_extractor import \
    FeatureVectorExtractor
from track_analysis.components.track_analysis.features.key_extraction.preprocessing.note_extraction.note_extractor import \
    NoteExtractor
from track_analysis.components.track_analysis.features.key_extraction.preprocessing.note_extraction.notes.note_event_builder import \
    NoteEvent
from track_analysis.components.track_analysis.features.key_extraction.profile_generation.model.key_profile import \
    KeyProfile
from track_analysis.components.track_analysis.features.key_extraction.utils.audio_loader import AudioLoader
from track_analysis.components.track_analysis.features.key_extraction.utils.beat_detector import BeatDetector


class SingleTrackProfiler:
    def __init__(
            self,
            audio_loader: AudioLoader,
            beat_detector: BeatDetector,
            note_extractor: NoteExtractor,
            feature_extractor: FeatureVectorExtractor,
    ):
        self._audio_loader = audio_loader
        self._beat_detector = beat_detector
        self._note_extractor = note_extractor
        self._feature_extractor = feature_extractor

    def profile(self, track_path: Path, track_key_label: str) -> KeyProfile:
        tonic, mode = parse_label(track_key_label)

        # 2) load audio, detect tempo, extract notes, then feature‐vector
        audio_samples, sr = self._audio_loader.load(track_path)
        tempo, _, _ = self._beat_detector.detect(audio_samples, sr)
        notes: List[NoteEvent] = self._note_extractor.extract(
            audio_samples, sr, tempo, visualize=False
        )
        vec = self._feature_extractor.extract_features_from_note_events(notes)

        # 3) wrap that 12‐dim vector into a KeyProfile
        return KeyProfile(tonic=tonic, mode=mode, vectors=[vec])
