import time
from pathlib import Path
from typing import Tuple, List

import numpy as np
from matplotlib.colors import ListedColormap

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import CACHE_DIRECTORY
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.main_extraction_pipeline.extract_harmonic import \
    HarmonicExtractor
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.main_extraction_pipeline.frequency_to_midi import \
    FrequencyToMidi
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.main_extraction_pipeline.magnitude_spectogram_extractor import \
    MagnitudeSpectogramExtractor
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.main_extraction_pipeline.midi_to_pitch_classes_converter import \
    MidiToPitchClassesConverter
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.main_extraction_pipeline.normalized_pitch_classes_cleaner import \
    NormalizedPitchClassesCleaner
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.main_extraction_pipeline.pitch_classes_normalizer import \
    PitchClassesNormalizer
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.main_extraction_pipeline.spectral_peak_extractor import \
    SpectralPeakExtractor
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.note_event_builder import \
    NoteEventBuilder, NoteEvent
from track_analysis.components.track_analysis.features.key_extraction.segmentation.audio_segmenter import AudioSegmenter
from track_analysis.components.track_analysis.features.key_extraction.segmentation.model.segmentation_result import \
    SegmentationResult
from track_analysis.components.track_analysis.features.key_extraction.utils.audio_loader import AudioLoader
from track_analysis.components.track_analysis.features.key_extraction.visualization.chroma_visualizer import \
    ChromaVisualizer
from track_analysis.components.track_analysis.features.key_extraction.visualization.model.config import \
    VisualizationConfig
from track_analysis.components.track_analysis.features.key_extraction.visualization.renderers.pianoroll_renderer import \
    PianoRollRenderer
from track_analysis.components.track_analysis.features.key_extraction.visualization.renderers.spectogram_renderer import \
    SpectrogramRenderer
from track_analysis.components.track_analysis.features.key_extraction.visualization.renderers.waveform_renderer import \
    WaveformRenderer


class NoteExtractor:
    """Pipeline orchestrator for extracting note features from a track file."""

    def __init__(self, logger: HoornLogger, subdivisions_per_beat: int = 2, hop_length_samples: int = 512, n_fft: int = 2048):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._logger.trace("Successfully initialized.", separator=self._separator)

        cache_dir: Path = CACHE_DIRECTORY
        self._hop_length_samples: int = hop_length_samples
        self._n_fft: int = n_fft
        self._audio_segmenter = AudioSegmenter(logger, cache_dir, self._hop_length_samples, subdivisions_per_beat=subdivisions_per_beat)

        self._audio_loader: AudioLoader = AudioLoader(logger, cache_dir / "audio loading")
        self._harmonic_extractor: HarmonicExtractor = HarmonicExtractor(logger, cache_dir / "harmonic extraction", hop_length_samples=self._hop_length_samples, n_fft=n_fft)
        self._magnitude_spec_extractor: MagnitudeSpectogramExtractor = MagnitudeSpectogramExtractor(logger, cache_dir / "magnitude spectrogram extraction", n_fft=self._n_fft, hop_length=self._hop_length_samples)
        self._spectral_peak_extractor: SpectralPeakExtractor = SpectralPeakExtractor(logger, cache_dir / "spectral peak extraction", min_frequency_hz=50, max_frequency_hz=2000, hop_length_samples=hop_length_samples, n_fft=self._n_fft)
        self._frequency_to_midi_converter: FrequencyToMidi = FrequencyToMidi(logger, cache_dir / "frequency to midi conversion")
        self._midi_to_pitch_classes_converter: MidiToPitchClassesConverter = MidiToPitchClassesConverter(logger, cache_dir / "midi to pitch classes conversion")
        self._pitch_classes_normalizer: PitchClassesNormalizer = PitchClassesNormalizer(logger, cache_dir / "pitch classes normalization conversion")
        self._normalized_pitch_classes_cleaner: NormalizedPitchClassesCleaner = NormalizedPitchClassesCleaner(logger, cache_dir / "normalized pitch-classes cleaning conversion")
        self._note_event_builder: NoteEventBuilder = NoteEventBuilder(logger)

        self._visualizer: ChromaVisualizer = ChromaVisualizer(self._hop_length_samples, logger)

    def extract(self, path: Path, time_signature: Tuple[int, int], min_segment_level: int, visualize: bool = False) -> Tuple[List[NoteEvent], SegmentationResult]:
        audio_samples_raw, sample_rate = self._audio_loader.load(path)
        segmentation_results, tempo = self._audio_segmenter.get_segments(audio_samples_raw, sample_rate, time_signature, min_segment_level)
        harmonic, percussive = self._harmonic_extractor.extract_harmonic(audio_samples_raw, sample_rate, tempo)
        harmonic_spec = self._magnitude_spec_extractor.extract_magnitude_spectogram(harmonic)
        frequencies, magnitudes = self._spectral_peak_extractor.extract_spectral_peaks(harmonic_spec, sample_rate)
        midi = self._frequency_to_midi_converter.convert(frequencies, magnitudes)
        pitch_classes = self._midi_to_pitch_classes_converter.convert(midi)
        normalized = self._pitch_classes_normalizer.normalize_pitch_classes(pitch_classes)
        cleaned_binary, cleaned_chroma = self._normalized_pitch_classes_cleaner.clean(normalized, audio_samples_raw, self._n_fft, self._hop_length_samples, sample_rate)
        note_events = self._note_event_builder.build_note_events(cleaned_binary, midi, self._hop_length_samples, sample_rate)

        if visualize:
            self._visualize([audio_samples_raw, harmonic, percussive, harmonic_spec, frequencies, magnitudes, midi, pitch_classes, normalized, cleaned_binary, cleaned_chroma], sample_rate)

        return note_events, segmentation_results

    def _visualize(self, visualizations: List[np.ndarray], sample_rate: int):
        visualization_cache = CACHE_DIRECTORY / "visualization"

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
                                    renderer=PianoRollRenderer(cmap='inferno', cache_dir=visualization_cache)),
                VisualizationConfig(data=pitch_classes, title="Pitch Classes (raw)", x_label="Time (s)", y_label="Pitch Classes", color_label="dB",
                                    renderer=PianoRollRenderer(cmap='inferno', cache_dir=visualization_cache, num_ticks=12)),
                VisualizationConfig(data=normalized, title="Pitch Classes (normalized L2)", x_label="Time (s)", y_label="Pitch Classes", color_label="dB",
                                    renderer=PianoRollRenderer(cmap='inferno', cache_dir=visualization_cache, num_ticks=12)),
                VisualizationConfig(data=cleaned, title="Pitch Classes (binary)", x_label="Time (s)", y_label="Pitch Classes", color_label="on/off",
                                    renderer=PianoRollRenderer(cmap=ListedColormap(["#000000", "#FFD700"]), cache_dir=visualization_cache, num_ticks=12, min_v=0, max_v=1, convert_to_db=False)),
                VisualizationConfig(data=cleaned_chroma, title="Pitch Classes (binary->chroma)", x_label="Time (s)", y_label="Pitch Classes", color_label="dB",
                                    renderer=PianoRollRenderer(cmap='inferno', cache_dir=visualization_cache, num_ticks=12)),
            ],
            sample_rate=sample_rate,
            combined=True
        )
