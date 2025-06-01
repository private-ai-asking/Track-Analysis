import pprint
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.key_extraction.preprocessing.note_extraction.note_extractor import \
    NoteExtractor


NOTE_TO_PC = {
    "C": 0,  "C#": 1, "Db": 1,
    "D": 2,  "D#": 3, "Eb": 3,
    "E": 4,  "Fb": 4, "E#": 5,
    "F": 5,  "F#": 6, "Gb": 6,
    "G": 7,  "G#": 8, "Ab": 8,
    "A": 9,  "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11
}


class ProfileCreator:
    """
    Used to compute a profile for Mode based on X amount of tracks. The more tracks, the more accurate.
    """

    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator = self.__class__.__name__

        self._note_extractor: NoteExtractor = NoteExtractor(logger)

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def compute_profile(self, corpus_csv_path: Path, alpha: float = 0.6):
        self._logger.debug(f"Starting compute_profile with corpus csv path: '{corpus_csv_path}'", separator=self._separator)

        # Read CSV
        try:
            csv: pd.DataFrame = pd.read_csv(corpus_csv_path)
        except Exception as e:
            self._logger.error(f"Failed to read CSV at '{corpus_csv_path}': {e}", separator=self._separator)
            raise

        num_tracks = len(csv)
        self._logger.info(f"Loaded corpus CSV with {num_tracks} tracks.", separator=self._separator)

        # Tonal: (Pitch Class: Present Count)
        number_present_notes: Dict[str, List[Dict[int, int]]] = {}

        for path, principal_key, _ in csv.itertuples(index=False):
            self._logger.debug(f"Processing track path: '{path}' with principal key: '{principal_key}'", separator=self._separator)

            principal_key_str = str(principal_key)
            track_path: Path = Path(path)

            # Extract tonal (note name part before space)
            tonal = principal_key_str.split(maxsplit=1)[0]
            self._logger.trace(f"Extracted tonal: '{tonal}' from principal key: '{principal_key_str}'", separator=self._separator)

            # Extract notes from the track
            try:
                note_events, _ = self._note_extractor.extract(track_path, min_segment_level=4, visualize=False)
            except Exception as e:
                self._logger.error(f"Note extraction failed for '{track_path}': {e}", separator=self._separator)
                continue

            num_notes = len(note_events)
            self._logger.debug(f"Extracted {num_notes} note events for '{track_path}'", separator=self._separator)

            # Initialize present notes counts
            present_notes = {pc: 0 for pc in range(12)}

            # Accumulate duration * energy per pitch class
            for note in note_events:
                present_notes[note.pitch_class] += (note.duration_seconds * note.total_energy)

            total_energy_sum = sum(present_notes.values())
            self._logger.trace(f"Total summed note energy for '{track_path}': {total_energy_sum}", separator=self._separator)

            # Group by tonal
            if tonal in number_present_notes:
                number_present_notes[tonal].append(present_notes)
                self._logger.debug(f"Appended present_notes to tonal group '{tonal}'. Group size now: {len(number_present_notes[tonal])}", separator=self._separator)
            else:
                number_present_notes[tonal] = [present_notes]
                self._logger.debug(f"Created new tonal group '{tonal}' with first entry.", separator=self._separator)

        num_tonals = len(number_present_notes)
        self._logger.info(f"Collected present notes for {num_tonals} tonals.", separator=self._separator)

        merged_to_c: List[Dict[int, int]] = []

        for tonal, list_of_dicts in number_present_notes.items():
            self._logger.debug(f"Transposing {len(list_of_dicts)} tracks for tonal '{tonal}'", separator=self._separator)
            if tonal not in NOTE_TO_PC:
                self._logger.error(f"Unrecognized tonal key: '{tonal}'", separator=self._separator)
                raise ValueError(f"Unrecognized tonal key: {tonal!r}")

            tonic_pc = NOTE_TO_PC[tonal]
            self._logger.trace(f"Tonic pitch class for tonal '{tonal}': {tonic_pc}", separator=self._separator)

            for track_dict in list_of_dicts:
                transposed_dict: Dict[int, int] = {i: 0 for i in range(12)}

                # For each original pitch‐class count, shift so that `tonic_pc` → 0 (C)
                for old_pc, count in track_dict.items():
                    new_pc = (old_pc - tonic_pc) % 12
                    transposed_dict[new_pc] += count

                merged_to_c.append(transposed_dict)

        num_merged = len(merged_to_c)
        self._logger.info(f"Merged and transposed counts into C for {num_merged} track entries.", separator=self._separator)

        # Step 2: For each pitch class 0–11, collect all counts across the merged list and compute the median
        median_profile: Dict[int, float] = {}  # type: ignore
        for pc in range(12):
            all_counts_for_pc = [track_dict.get(pc, 0) for track_dict in merged_to_c]
            self._logger.trace(f"Computing median for pitch class {pc}, with {len(all_counts_for_pc)} values.", separator=self._separator)
            median_profile[pc] = float(np.median(all_counts_for_pc))  # type: ignore

        self._logger.debug(f"Raw median profile (before normalization):\n{pprint.pformat(median_profile)}", separator=self._separator)

        # Step 3: Apply non-linear compression (power transform) to match dynamic range of K&S templates
        self._logger.info(f"Applying power transform with alpha = {alpha} to median profile.", separator=self._separator)
        compressed_profile = {pc: value ** alpha for pc, value in median_profile.items()}
        self._logger.trace(f"Compressed profile (before normalization):\n{pprint.pformat(compressed_profile)}", separator=self._separator)

        # Step 4: Normalize the compressed profile so that the sum of all bins = 38.5
        total_compressed_sum = sum(compressed_profile.values())
        self._logger.info(f"Sum of compressed profile values: {total_compressed_sum}", separator=self._separator)

        if total_compressed_sum == 0:
            self._logger.error("Sum of compressed profile is zero; cannot normalize.", separator=self._separator)
            raise RuntimeError("Sum of compressed profile is zero; cannot normalize.")

        scaling_factor = 38.5 / total_compressed_sum
        self._logger.info(f"Calculated scaling factor after compression: {scaling_factor}", separator=self._separator)

        normalized_profile: Dict[int, float] = {
            pc: compressed_value * scaling_factor
            for pc, compressed_value in compressed_profile.items()
        }

        values = [str(v) for _, v in normalized_profile.items()]
        values_rounded = [str(round(v, 2)) for _, v in normalized_profile.items()]

        self._logger.info(f"Normalized Profile (final):\n{pprint.pformat(normalized_profile)}", separator=self._separator)
        self._logger.info(f"Normalized values (precise): [{', '.join(values)}]", separator=self._separator)
        self._logger.info(f"Normalized values (rounded): [{', '.join(values_rounded)}]", separator=self._separator)

        return normalized_profile
