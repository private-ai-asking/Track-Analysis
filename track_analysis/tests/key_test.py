from pathlib import Path
from typing import Tuple

import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.testing import TestInterface
from track_analysis.components.track_analysis.features.key_extraction.key_extraction.lof_feature_transformer import \
    LOFFeatureTransformer
from track_analysis.components.track_analysis.features.key_extraction.key_extraction.run_length_merger import \
    RunLengthMerger
from track_analysis.components.track_analysis.features.key_extraction.key_extraction.sequence_optimizer import \
    SequenceOptimizer
from track_analysis.components.track_analysis.features.key_extraction.key_extraction.template_scorer import \
    TemplateScorer
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.note_extractor import \
    NoteExtractor
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.segment_profiler import \
    SegmentProfiler


class KeyProgressionTest(TestInterface):
    def __init__(self, logger: HoornLogger, modulation_penalty: float = 6.0):
        super().__init__(logger, is_child=True)
        self._separator = 'KeyProgressionTest'

        # build music templates
        ionian = np.array([5.0,2.0,3.5,2.0,4.5,4.0,2.0,4.5,2.0,3.5,1.5,4.0])
        aeolian = np.array([5.0,2.0,3.5,4.5,2.0,4.0,2.0,4.5,3.5,2.0,1.5,4.0])
        dorian = np.array([5.0,2.0,3.5,4.5,2.0,4.0,2.0,4.5,2.0,3.5,1.5,4.0])
        modes = {'Ionian (Major)': ionian, 'Aeolian (Minor)': aeolian, 'Dorian (Minor)': dorian}
        tonics = ['C','G','D','A','E','B','F#','C#','G#','D#','A#','F']

        # map and normalize templates
        transformer = LOFFeatureTransformer()
        _templates = {}
        for mode_name, arr in modes.items():
            base = transformer.transform(arr)
            for shift, tonic in enumerate(tonics):
                name = f"{tonic} {mode_name}"
                tmpl = np.roll(base, shift)
                norm = tmpl / tmpl.sum()
                _templates[name] = norm

        self._scorer = TemplateScorer(_templates)
        penalty_scaled = modulation_penalty * 2.0 / ionian.sum()
        self._optimizer = SequenceOptimizer(penalty_scaled)
        self._merger = RunLengthMerger()

        self._note_extractor = NoteExtractor(logger, subdivisions_per_beat=2, hop_length_samples=512)
        self._segment_profiler = SegmentProfiler(logger)

    def test(
            self,
            file_path: Path,
            time_signature: Tuple[int,int] = (4,4),
            segment_beat_level: int = 3
    ) -> None:
        if not file_path.is_file():
            self._logger.error(f"File not found: {file_path}", separator=self._separator)
            return
        self._logger.info(f"Analyzing file: {file_path}", separator=self._separator)

        note_events, seg_res = self._note_extractor.extract(
            file_path, time_signature, segment_beat_level, visualize=False
        )
        segments = self._segment_profiler.profile_segments(seg_res, note_events)

        # build feature vectors
        feature_vecs = []
        times = []
        for seg in segments:
            hist = np.zeros(12)
            for e in seg.segment_notes:
                hist[e.pitch_class] += 1
            fv = LOFFeatureTransformer().transform(hist)
            total = fv.sum()
            fv = fv/total if total>0 else fv
            feature_vecs.append(fv)
            times.append((seg.segment_start_seconds, seg.segment_end_seconds))

        # scoring + optimization
        raw_scores, keys = self._scorer.score(feature_vecs)
        path = self._optimizer.solve(raw_scores)
        runs = self._merger.merge(times, path.tolist(), keys)

        # logging
        for run in runs:
            dur = run['end'] - run['start']
            self._logger.info(
                f"[segment {run['idx']}] {self._format_time(run['start'])} -> {self._format_time(run['end'])} "
                f"({dur:.2f}s) => {run['state']}",
                separator=self._separator
            )
        self._logger.info("Key progression complete.", separator=self._separator)

    @staticmethod
    def _format_time(seconds: float) -> str:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}:{secs:05.2f}"
