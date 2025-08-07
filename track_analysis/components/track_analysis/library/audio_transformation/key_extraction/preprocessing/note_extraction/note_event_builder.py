from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pprint
import librosa

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import VERBOSE


@dataclass(frozen=True)
class NoteEvent:
    pitch_class: int
    midi_pitch: int
    onset_seconds: float
    offset_seconds: float
    duration_seconds: float
    mean_energy: float
    max_energy: float
    total_energy: float
    raw_energy: np.ndarray


class NoteEventBuilder:
    def __init__(self, logger: HoornLogger, hop_length: int = 512):
        self._logger = logger
        self._separator = self.__class__.__name__
        self._hop_length = hop_length
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def build_note_events(
            self,
            cleaned_mask:   np.ndarray,
            midi_map:       np.ndarray,
            sr:             int
    ) -> List[NoteEvent]:
        """
        Turn a 12×T cleaned_mask + 128×T midi_map → List[NoteEvent] with true midi_pitch
        and raw_energy per frame so you can aggregate in any way downstream.
        """
        events: List[NoteEvent] = []
        n_pc, T = cleaned_mask.shape
        midi_to_pc = np.arange(128) % 12
        frame_duration = self._hop_length / sr

        for pc in range(n_pc):
            mask = cleaned_mask[pc, :]
            padded = np.concatenate([[0], mask, [0]])  # type: ignore
            changes = np.diff(padded.astype(int))
            on_frames  = np.where(changes == +1)[0]
            off_frames = np.where(changes == -1)[0]

            for onset_f, offset_f in zip(on_frames, off_frames):
                onset_sec  = onset_f  * frame_duration
                offset_sec = offset_f * frame_duration
                duration   = offset_sec - onset_sec
                frames = np.arange(onset_f, offset_f)
                if frames.size == 0:
                    continue

                snippet = midi_map[:, frames]
                pc_mask = (midi_to_pc[:, None] == pc)
                snippet_pc = snippet * pc_mask
                best_midi_per_frame = snippet_pc.argmax(axis=0)
                raw_energy = snippet_pc[best_midi_per_frame, np.arange(best_midi_per_frame.shape[0])]
                total_e = float(raw_energy.sum() * frame_duration)

                mean_e = float(raw_energy.mean())
                max_e  = float(raw_energy.max())
                sum_energy_by_midi = np.bincount(
                    best_midi_per_frame,
                    weights=raw_energy,
                    minlength=128
                )
                midi_pitch = int(sum_energy_by_midi.argmax()) if sum_energy_by_midi.sum() > 0 else pc

                events.append(NoteEvent(
                    pitch_class=pc,
                    midi_pitch=midi_pitch,
                    onset_seconds=onset_sec,
                    offset_seconds=offset_sec,
                    duration_seconds=duration,
                    mean_energy=mean_e,
                    max_energy=max_e,
                    raw_energy=raw_energy,
                    total_energy=total_e,
                ))

        if VERBOSE:
            self._log_summary(events)

        return events

    @staticmethod
    def _format_stats(values: np.ndarray) -> Dict[str, float]:
        """
        Compute mean, median, and max for a numpy array of values.
        """
        return {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'max': float(np.max(values))
        }

    def _log_summary(self, events: List[NoteEvent]) -> None:
        summary: Dict[int, Any] = {}
        for pc in sorted({e.pitch_class for e in events}):
            evs = [e for e in events if e.pitch_class == pc]
            count = len(evs)
            idxs = np.linspace(0, count - 1, num=min(5, count), dtype=int)
            sampled = [evs[i] for i in idxs]

            # Truncate raw_energy on each sampled event
            raw_events = []
            for e in sampled:
                re = e.raw_energy
                L = re.shape[0]
                if L <= 5:
                    small = re
                else:
                    sel = np.linspace(0, L - 1, num=5, dtype=int)
                    small = re[sel]
                mini = {
                    'pitch_class': e.pitch_class,
                    'midi_pitch' : e.midi_pitch,
                    'duration_s' : e.duration_seconds,
                    'raw_energy' : small.tolist()
                }
                raw_events.append(mini)

            durations   = np.array([e.duration_seconds for e in evs])
            energies_lin= np.array([e.mean_energy      for e in evs])
            energies_db = librosa.amplitude_to_db(energies_lin, ref=1.0)

            summary[pc] = {
                'raw_events'          : raw_events,
                'duration_stats'      : self._format_stats(durations),
                'energy_linear_stats' : self._format_stats(energies_lin),
                'energy_db_stats'     : self._format_stats(energies_db),
                'count'               : count
            }

        formatted = pprint.pformat(summary)
        self._logger.debug(f"NoteEvent summary per pitch class:\n{formatted}",
                           separator=self._separator)

# TODO - see if we can cache here
