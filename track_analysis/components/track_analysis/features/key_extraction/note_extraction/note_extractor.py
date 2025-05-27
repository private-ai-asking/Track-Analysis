from pathlib import Path
from typing import Tuple, Dict, List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.key_extraction.audio_segmenter import AudioSegmenter
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.chroma_extractor import \
    ChromaExtractor
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.decibel_converter import \
    DecibelConverter
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.mask_filter import MaskFilter
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.model.note_event import NoteEvent
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.note_event_builder import \
    NoteEventBuilder
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.thresholding import \
    BinaryMaskGenerator, OtsuThresholding
from track_analysis.components.track_analysis.features.key_extraction.segmentation.model.segmentation_result import \
    SegmentationResult
from track_analysis.components.track_analysis.features.key_extraction.utils.audio_loader import AudioLoader


class NoteExtractor:
    def __init__(self, logger: HoornLogger, hop_length_ms: int = 512, subdivisions_per_beat: int = 2):
        self._hop_length_ms: int = hop_length_ms

        self._loader    = AudioLoader(logger)
        self._chroma_extractor  = ChromaExtractor(hop_length=hop_length_ms, logger=logger)
        self._audio_segmenter: AudioSegmenter = AudioSegmenter(logger, subdivisions_per_beat=subdivisions_per_beat)
        self._db_converter    = DecibelConverter()
        self._binary_mask_generator    = BinaryMaskGenerator(OtsuThresholding(), logger=logger)
        self._mask_filter  = MaskFilter(logger)
        self._event_note_builder = NoteEventBuilder(hop_length=hop_length_ms, logger=logger)

    def extract(self, path: Path, time_signature: Tuple[int, int], min_segment_beat_level: int) -> Tuple[Dict[int, List[NoteEvent]], SegmentationResult]:
        audio, sample_rate = self._loader.load(path)
        segment_results, tempo = self._audio_segmenter.get_segments(audio, sample_rate, time_signature, min_segment_beat_level)
        chroma = self._chroma_extractor.extract(audio, sample_rate)
        chroma_db = self._db_converter.to_db(chroma)
        binary_mask = self._binary_mask_generator.binarize(chroma_db)
        clean_mask = self._mask_filter.filter(binary_mask, self._hop_length_ms, sample_rate, tempo, min_segment_beat_level)
        return self._event_note_builder.build(clean_mask, sample_rate), segment_results
