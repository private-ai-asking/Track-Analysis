import gc
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from matplotlib.colors import ListedColormap

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import EXPENSIVE_CACHE_DIRECTORY
from track_analysis.components.track_analysis.features.core.caching.cached_operations.frequency_to_midi import FrequencyToMidi
from track_analysis.components.track_analysis.features.core.caching.cached_operations.harmonic import HarmonicExtractor
from track_analysis.components.track_analysis.features.core.caching.cached_operations.magnitude_spectogram import \
    MagnitudeSpectrogramExtractor
from track_analysis.components.track_analysis.features.core.caching.cached_operations.midi_to_pitch import MidiToPitchClassesConverter
from track_analysis.components.track_analysis.features.core.caching.cached_operations.pitch_class_cleaner import \
    NormalizedPitchClassesCleaner
from track_analysis.components.track_analysis.features.core.caching.cached_operations.pitch_class_normalizer import \
    PitchClassesNormalizer
from track_analysis.components.track_analysis.features.core.caching.cached_operations.spectral_peak import SpectralPeakExtractor
from track_analysis.components.track_analysis.features.key_extraction.feature.visualization.chroma_visualizer import \
    ChromaVisualizer
from track_analysis.components.track_analysis.features.key_extraction.feature.visualization.model.config import \
    VisualizationConfig
from track_analysis.components.track_analysis.features.key_extraction.feature.visualization.renderers.pianoroll_renderer import \
    PianoRollRenderer
from track_analysis.components.track_analysis.features.key_extraction.feature.visualization.renderers.spectogram_renderer import \
    SpectrogramRenderer
from track_analysis.components.track_analysis.features.key_extraction.feature.visualization.renderers.waveform_renderer import \
    WaveformRenderer
from track_analysis.components.track_analysis.features.key_extraction.preprocessing.note_extraction.notes.note_event_builder import \
    NoteEventBuilder, NoteEvent

@dataclass
class NoteExtractionResult:
    notes: List[NoteEvent]
    harmonic: np.ndarray = None
    percussive: np.ndarray = None


class NoteExtractor:
    """Pipeline orchestrator for extracting note features from a track file."""

    def __init__(self, logger: HoornLogger, hop_length_samples: int = 512, n_fft: int = 2048):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._logger.trace("Successfully initialized.", separator=self._separator)

        self._hop_length_samples: int = hop_length_samples
        self._n_fft: int = n_fft

        self._harmonic_extractor: HarmonicExtractor = HarmonicExtractor(logger, hop_length=self._hop_length_samples, n_fft=n_fft)
        self._magnitude_spec_extractor: MagnitudeSpectrogramExtractor = MagnitudeSpectrogramExtractor(logger, n_fft=self._n_fft, hop_length=self._hop_length_samples)
        self._spectral_peak_extractor: SpectralPeakExtractor = SpectralPeakExtractor(logger, min_frequency_hz=50, max_frequency_hz=2000, hop_length_samples=hop_length_samples, n_fft=self._n_fft)
        self._frequency_to_midi_converter: FrequencyToMidi = FrequencyToMidi(logger, n_fft=n_fft, hop_length=self._hop_length_samples)
        self._midi_to_pitch_classes_converter: MidiToPitchClassesConverter = MidiToPitchClassesConverter(logger)
        self._pitch_classes_normalizer: PitchClassesNormalizer = PitchClassesNormalizer(logger)
        self._normalized_pitch_classes_cleaner: NormalizedPitchClassesCleaner = NormalizedPitchClassesCleaner(logger)
        self._note_event_builder: NoteEventBuilder = NoteEventBuilder(logger)

        self._visualizer: ChromaVisualizer = ChromaVisualizer(self._hop_length_samples, logger)

    def extract(self, file_path: Path, audio_samples_raw: np.ndarray, sample_rate: int, track_tempo: float, visualize: bool = False) -> NoteExtractionResult:
        harmonic, percussive = self._harmonic_extractor.extract_harmonic(file_path=file_path, audio=audio_samples_raw, sample_rate=sample_rate, tempo_bpm=track_tempo)
        harmonic_spec = self._magnitude_spec_extractor.extract(file_path=file_path, audio=harmonic, start_sample=0, end_sample=harmonic.shape[0])
        frequencies, magnitudes = self._spectral_peak_extractor.extract_spectral_peaks(file_path=file_path, spectral_data=harmonic_spec, start_sample=0, end_sample=harmonic_spec.shape[0], sample_rate=sample_rate)
        midi = self._frequency_to_midi_converter.convert(file_path=file_path, start_sample=0, end_sample=frequencies.shape[0], frequencies=frequencies, magnitudes=magnitudes, sample_rate=sample_rate)
        pitch_classes = self._midi_to_pitch_classes_converter.convert(file_path, midi)
        normalized = self._pitch_classes_normalizer.normalize_pitch_classes(file_path, pitch_classes)
        cleaned_binary, cleaned_chroma = self._normalized_pitch_classes_cleaner.clean(file_path, normalized, audio_samples_raw, self._n_fft, self._hop_length_samples, sample_rate)
        note_events = self._note_event_builder.build_note_events(cleaned_binary, midi, self._hop_length_samples, sample_rate)

        if visualize:
            self._visualize([audio_samples_raw, harmonic, percussive, harmonic_spec, frequencies, magnitudes, midi, pitch_classes, normalized, cleaned_binary, cleaned_chroma], sample_rate)

        del harmonic_spec
        del frequencies
        del magnitudes
        del midi
        del pitch_classes
        del normalized
        del cleaned_binary
        del cleaned_chroma

        gc.collect()

        return NoteExtractionResult(note_events, harmonic, percussive)

    def _visualize(self, visualizations: List[np.ndarray], sample_rate: int):
        visualization_cache = EXPENSIVE_CACHE_DIRECTORY / "visualization"

        audio_samples_raw = visualizations[0]
        harmonic = visualizations[1]
        percussive = visualizations[2]
        harmonic_spec = visualizations[3]
        frequencies = visualizations[4]
        magnitudes = visualizations[5]
        midi = visualizations[6]
        pitch_classes = visualizations[7]
        normalized = visualizations[8]
        cleaned = visualizations[9]
        cleaned_chroma = visualizations[10]

        self._visualizer.visualize(
            configs=[
                VisualizationConfig(data=audio_samples_raw, title="Audio Samples (Raw)", x_label="Time (s)", y_label="Amplitude (a.u.)", renderer=WaveformRenderer(sample_rate=sample_rate, mode="Full", cache_dir=visualization_cache)),
                VisualizationConfig(data=audio_samples_raw, title="Audio Samples (Envelope)", x_label="Time (s)", y_label="Amplitude (a.u.)", renderer=WaveformRenderer(sample_rate=sample_rate, cache_dir=visualization_cache)),
                VisualizationConfig(data=harmonic, title="Harmonic (raw through viz)", x_label="Time (s)", y_label="Frequency (Hz)", color_label="dB",
                                    renderer=SpectrogramRenderer(sample_rate=sample_rate, hop_length=self._hop_length_samples, n_fft=self._n_fft, cache_dir=visualization_cache)),
                VisualizationConfig(data=percussive, title="Percussive", x_label="Time (s)", y_label="Frequency (Hz)", color_label="dB",
                                    renderer=SpectrogramRenderer(sample_rate=sample_rate, hop_length=self._hop_length_samples, n_fft=self._n_fft, cache_dir=visualization_cache)),
                VisualizationConfig(data=harmonic_spec, title="Harmonic (Extracted Spectogram)", x_label="Time (s)", y_label="Frequency (Hz)", color_label="dB",
                                    renderer=SpectrogramRenderer(sample_rate=sample_rate, hop_length=self._hop_length_samples, n_fft=self._n_fft, cache_dir=visualization_cache)),
                VisualizationConfig(data=frequencies, title="Pitches", x_label="Time (s)", y_label="Frequency (Hz)", color_label="dB",
                                    renderer=SpectrogramRenderer(sample_rate=sample_rate, hop_length=self._hop_length_samples, n_fft=self._n_fft, cache_dir=visualization_cache)),
                VisualizationConfig(data=magnitudes, title="Magnitudes", x_label="Time (s)", y_label="Frequency (Hz)", color_label="dB",
                                    renderer=SpectrogramRenderer(sample_rate=sample_rate, hop_length=self._hop_length_samples, n_fft=self._n_fft, cache_dir=visualization_cache)),
                VisualizationConfig(data=midi, title="Midi Notes", x_label="Time (s)", y_label="MIDI pitch", color_label="dB",
                                    renderer=PianoRollRenderer(cmap='inferno')),
                VisualizationConfig(data=pitch_classes, title="Pitch Classes (raw)", x_label="Time (s)", y_label="Pitch Classes", color_label="dB",
                                    renderer=PianoRollRenderer(cmap='inferno', num_ticks=12)),
                VisualizationConfig(data=normalized, title="Pitch Classes (normalized L2)", x_label="Time (s)", y_label="Pitch Classes", color_label="dB",
                                    renderer=PianoRollRenderer(cmap='inferno', num_ticks=12)),
                VisualizationConfig(data=cleaned, title="Pitch Classes (binary)", x_label="Time (s)", y_label="Pitch Classes", color_label="on/off",
                                    renderer=PianoRollRenderer(cmap=ListedColormap(["#000000", "#FFD700"]), num_ticks=12, min_v=0, max_v=1, convert_to_db=False)),
                VisualizationConfig(data=cleaned_chroma, title="Pitch Classes (binary->chroma)", x_label="Time (s)", y_label="Pitch Classes", color_label="dB",
                                    renderer=PianoRollRenderer(cmap='inferno', num_ticks=12)),
            ],
            sample_rate=sample_rate,
            combined=True
        )
