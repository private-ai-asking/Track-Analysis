import pprint
import random
from typing import Dict, List

import librosa
import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import VERBOSE
from track_analysis.components.track_analysis.features.key_extraction.note_extraction.model.note_event import NoteEvent


class NoteEventBuilder:
    def __init__(self, hop_length: int, logger: HoornLogger):
        self.hop_length, self._logger = hop_length, logger
        self._separator: str = "NoteEventBuilder"

    def build(self, clean_mask: np.ndarray, sr: int) -> Dict[int, List[NoteEvent]]:
        times = librosa.frames_to_time(np.arange(clean_mask.shape[1]), sr=sr, hop_length=self.hop_length)
        events = {pc: [] for pc in range(clean_mask.shape[0])}
        for pc in events:
            row = clean_mask[pc].astype(int)
            edges = np.diff(row, prepend=0, append=0)
            on_f = np.where(edges==1)[0]
            off_f= np.where(edges==-1)[0]
            for o,f in zip(on_f, off_f):
                events[pc].append(NoteEvent(pc, float(times[o]), float(times[f])))

        self._print_summary(events)

        return events

    def _print_summary(self, events: Dict) -> None:
        if VERBOSE:
            preview: Dict[int, List[NoteEvent]] = {}

            # noinspection PyPep8Naming
            MAX_PER_CLASS = 5
            for pc, evs in events.items():
                preview[pc] = random.sample(evs, min(MAX_PER_CLASS, len(evs)))
            summary = {
                pc: {
                    "total": len(events[pc]),
                    "preview": preview[pc]
                }
                for pc in range(12)
            }

            self._logger.debug(f"Note‐event summary:\n{pprint.pformat(summary)}",
                               separator=self._separator)
