from dataclasses import dataclass
from typing import List

import numpy as np
import pprint

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import VERBOSE
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.preprocessing.note_extraction.notes.note_event_builder import NoteEvent
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.preprocessing.segmentation.model.segmentation_result import SegmentationResult


@dataclass(frozen=True)
class SegmentNote:
    pitch_class: int
    note_start_seconds_in_segment: float
    note_end_seconds_in_segment: float
    note_duration_seconds_in_segment: float
    raw_energy_in_segment: np.ndarray
    mean_energy_in_segment: float
    median_energy_in_segment: float
    max_energy_in_segment: float
    total_energy_in_segment: float


@dataclass(frozen=True)
class Segment:
    segment_start_seconds: float
    segment_end_seconds: float
    segment_duration_seconds: float
    segment_notes: List[SegmentNote]

    @property
    def num_pitch_classes(self) -> int:
        encountered = set(e.pitch_class for e in self.segment_notes)
        return len(encountered)

    @property
    def num_notes(self) -> int:
        return len(self.segment_notes)


class SegmentProfiler:
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def profile_segments(
            self,
            segmentation_results: SegmentationResult,
            note_events: List[NoteEvent],
            preview_n: int = 3,
            preview_x: int = 5
    ) -> List[Segment]:
        """
        For each segment, collect notes that intersect it and profile energy.
        Optionally preview a subset of segments and notes.
        """
        segments_out: List[Segment] = []

        for seg_idx, (seg_samples, seg_start, seg_dur) in enumerate(
                zip(
                    segmentation_results.segments,
                    segmentation_results.start_times,
                    segmentation_results.durations
                )
        ):
            seg_end = seg_start + seg_dur
            notes: List[SegmentNote] = []

            for e in note_events:
                ev_start, ev_end = e.onset_seconds, e.offset_seconds
                clip_start = max(ev_start, seg_start)
                clip_end   = min(ev_end, seg_end)
                if clip_end <= clip_start:
                    continue

                start_rel = clip_start - seg_start
                end_rel   = clip_end   - seg_start
                dur_rel   = end_rel - start_rel

                total_dur = ev_end - ev_start
                frames = e.raw_energy.shape[0]
                start_idx = int(np.floor((clip_start - ev_start) / total_dur * frames))
                end_idx   = int(np.ceil ((clip_end   - ev_start) / total_dur * frames))
                raw_seg   = e.raw_energy[start_idx:end_idx]

                mean_e = float(raw_seg.mean())
                med_e  = float(np.median(raw_seg))
                max_e  = float(raw_seg.max())
                total_e = float(raw_seg.sum())

                notes.append(
                    SegmentNote(
                        pitch_class=e.pitch_class,
                        note_start_seconds_in_segment=start_rel,
                        note_end_seconds_in_segment=end_rel,
                        note_duration_seconds_in_segment=dur_rel,
                        raw_energy_in_segment=raw_seg,
                        mean_energy_in_segment=mean_e,
                        median_energy_in_segment=med_e,
                        max_energy_in_segment=max_e,
                        total_energy_in_segment=total_e
                    )
                )

            segments_out.append(
                Segment(
                    segment_start_seconds=seg_start,
                    segment_end_seconds=seg_end,
                    segment_duration_seconds=seg_dur,
                    segment_notes=notes
                )
            )
            self._logger.debug(
                f"Segment {seg_idx}: {len(notes)} notes, "
                f"{segments_out[-1].num_pitch_classes} pitch classes",
                separator=self._separator
            )

        # Preview debug
        if VERBOSE:
            self._preview_debug(segments_out, preview_n, preview_x)

        # Stats of unique notes per segment
        uniques = np.array([seg.num_pitch_classes for seg in segments_out])
        stats = {
            'mean_unique_notes': float(np.mean(uniques)),
            'median_unique_notes': float(np.median(uniques)),
            'max_unique_notes': int(np.max(uniques))
        }
        self._logger.debug(
            f"Unique notes per segment stats:\n{pprint.pformat(stats)}",
            separator=self._separator
        )

        return segments_out

    def _preview_debug(
            self,
            segments: List[Segment],
            n_segments: int,
            x_notes: int
    ) -> None:
        """
        Print a preview of up to n_segments (evenly sampled) and for each, up to x_notes (evenly sampled).
        Shows total notes, unique pitch classes, and pformatted note details.
        """
        total_segs = len(segments)
        if total_segs == 0:
            return
        idxs = np.linspace(0, total_segs - 1, num=min(n_segments, total_segs), dtype=int)

        preview = []
        for i in idxs:
            seg = segments[i]
            notes = seg.segment_notes
            cnt_notes = len(notes)
            cnt_unique = seg.num_pitch_classes

            # sample notes
            if cnt_notes == 0:
                sampled = []
            else:
                note_idxs = np.linspace(0, cnt_notes - 1, num=min(x_notes, cnt_notes), dtype=int)
                sampled = [notes[j] for j in note_idxs]

            notes = [{
                'pitch': n.pitch_class,
                'start (s)': n.note_start_seconds_in_segment,
                'end (s)': n.note_end_seconds_in_segment,
                'duration (s)': n.note_duration_seconds_in_segment,
                'mean energy': n.mean_energy_in_segment,
                'median energy': n.median_energy_in_segment,
                'max energy': n.max_energy_in_segment
            } for n in sampled]

            preview.append({
                'segment_index': i,
                'segment_window': (seg.segment_start_seconds, seg.segment_end_seconds),
                'total_notes': cnt_notes,
                'unique_pitch_classes': cnt_unique,
                'sampled_notes': notes
            })

        out = pprint.pformat(preview, sort_dicts=False)
        self._logger.debug(f"Segment preview:\n{out}", separator=self._separator)
