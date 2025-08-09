from collections import defaultdict
from typing import List, Dict, Tuple

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.library.audio_transformation.key_extraction.profile_generation.model.key_profile import \
    KeyProfile


class KeyProfileMerger:
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator: str = self.__class__.__name__

    def group(self, key_profiles: List[KeyProfile]) -> Dict[Tuple[str, str], List[KeyProfile]]:
        grouped: Dict[Tuple[str, str], List[KeyProfile]] = defaultdict(list)

        for kp in key_profiles:
            key = (kp.tonic, kp.mode)
            grouped[key].append(kp)
            count = len(grouped[key])

            self._logger.debug(
                f'Appended 1 entry to the "{key[0]} {key[1]}" group. Total entries: {count}.',
                separator=self._separator
            )

        return dict(grouped)

    def merge(self, key_profiles: List[KeyProfile]) -> List[KeyProfile]:
        grouped = self.group(key_profiles)
        merged_profiles: List[KeyProfile] = []

        for (tonic, mode), profiles in grouped.items():
            combined_vectors: List[np.ndarray] = []
            for profile in profiles:
                combined_vectors.extend(profile.vectors)

            merged = KeyProfile(tonic=tonic, mode=mode, vectors=combined_vectors)
            self._logger.debug(
                f'Merged {len(profiles)} profiles into "{tonic} {mode}" with {len(combined_vectors)} total vectors.',
                separator=self._separator
            )
            merged_profiles.append(merged)

        return merged_profiles
