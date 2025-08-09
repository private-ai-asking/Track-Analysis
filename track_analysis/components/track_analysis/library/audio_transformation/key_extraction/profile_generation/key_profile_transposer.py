from typing import List, Dict

import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.profile_generation.model.key_profile import \
    KeyProfile

_SEMITONE_MAP: Dict[str, int] = {
    "C": 0,  "C#": 1, "Db": 1,
    "D": 2,  "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5,  "F#": 6, "Gb": 6,
    "G": 7,  "G#": 8, "Ab": 8,
    "A": 9,  "A#": 10, "Bb": 10,
    "B": 11
}


class KeyProfileTransposer:
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator: str = self.__class__.__name__

    def transpose_profiles_to_c(self, merged_profiles: List[KeyProfile]) -> List[KeyProfile]:
        """
        Transpose all merged KeyProfiles so that their tonic becomes C, then combine
        all vectors into one KeyProfile per mode (tonic='C').
        """
        mode_to_vectors: Dict[str, List[np.ndarray]] = {}

        for profile in merged_profiles:
            transposed = self._transpose_vectors_to_c(profile)
            self._accumulate_transposed_vectors(mode_to_vectors, profile.mode, transposed, profile.tonic)

        return self._build_c_mode_profiles(mode_to_vectors)

    def _transpose_vectors_to_c(self, profile: KeyProfile) -> List[np.ndarray]:
        """
        Take a KeyProfile and transpose each 12-length chromatic feature vector
        so that the tonic maps to C (i.e. roll by -offset).
        """
        tonic = profile.tonic
        if tonic not in _SEMITONE_MAP:
            self._logger.warning(f"Unknown tonic '{tonic}'", separator=self._separator)
            raise ValueError(f"Unknown tonic name: {tonic}")

        offset = _SEMITONE_MAP[tonic]
        transposed_list: List[np.ndarray] = []

        for vec in profile.vectors:
            if vec.ndim != 1 or vec.shape[0] != 12:
                raise ValueError(
                    f"Expected a 12-length 1D array for feature vectors, got shape {vec.shape}"
                )
            shifted = np.roll(vec, -offset)
            transposed_list.append(shifted)

        return transposed_list

    def _accumulate_transposed_vectors(
            self,
            mode_to_vectors: Dict[str, List[np.ndarray]],
            mode: str,
            transposed: List[np.ndarray],
            original_tonic: str
    ) -> None:
        """
        Add the transposed vectors under the given mode key in mode_to_vectors,
        logging whether this is the first addition or an append.
        """
        if mode not in mode_to_vectors:
            mode_to_vectors[mode] = transposed.copy()
            self._logger.debug(
                f"Initialized C-{mode} profile with {len(transposed)} vectors (shifted from {original_tonic}).",
                separator=self._separator
            )
        else:
            mode_to_vectors[mode].extend(transposed)
            total = len(mode_to_vectors[mode])
            self._logger.debug(
                f"Added {len(transposed)} more vectors to C-{mode} profile (shifted from {original_tonic}). "
                f"Total now: {total}.",
                separator=self._separator
            )

    def _build_c_mode_profiles(self, mode_to_vectors: Dict[str, List[np.ndarray]]) -> List[KeyProfile]:
        """
        Construct a list of KeyProfile objects, one per mode, all with tonic='C',
        collecting all accumulated vectors.
        """
        output_profiles: List[KeyProfile] = []

        for mode, vectors in mode_to_vectors.items():
            profile = KeyProfile(tonic="C", mode=mode, vectors=vectors.copy())
            self._logger.debug(
                f"Created final merged profile for C-{mode} with {len(vectors)} vectors.",
                separator=self._separator
            )
            output_profiles.append(profile)

        return output_profiles
