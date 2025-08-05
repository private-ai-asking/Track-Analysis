import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler
from track_analysis.components.track_analysis.features.core.caching.cached_operations.beat import BeatDetector
from track_analysis.components.track_analysis.features.key_extraction.feature.lof.lof_feature_transformer import (
    LOFFeatureTransformer,
)
from track_analysis.components.track_analysis.features.key_extraction.feature.vector.feature_vector_extractor import (
    FeatureVectorExtractor,
)
from track_analysis.components.track_analysis.features.key_extraction.preprocessing.note_extraction.note_extractor import (
    NoteExtractor,
)
from track_analysis.components.track_analysis.features.key_extraction.profile_generation.key_profile_merger import (
    KeyProfileMerger,
)
from track_analysis.components.track_analysis.features.key_extraction.profile_generation.key_profile_transposer import (
    KeyProfileTransposer,
)
from track_analysis.components.track_analysis.features.key_extraction.profile_generation.model.key_profile import (
    KeyProfile,
)
from track_analysis.components.track_analysis.features.key_extraction.profile_generation.single_track_profiler import (
    SingleTrackProfiler,
)


class ProfileGenerator:
    def __init__(self, logger: HoornLogger, audio_loader: AudioFileHandler, template_profile_normalized_to: int = 100, num_workers: int = 8):
        self._logger = logger
        self._separator = self.__class__.__name__

        # build all the components exactly once
        self._audio_loader: AudioFileHandler = audio_loader
        self._beat_detector: BeatDetector = BeatDetector(logger)
        self._note_extractor: NoteExtractor = NoteExtractor(logger)
        self._feature_extractor: FeatureVectorExtractor = FeatureVectorExtractor(
            logger, LOFFeatureTransformer()
        )
        self._track_profiler = SingleTrackProfiler(
            audio_loader=self._audio_loader,
            beat_detector=self._beat_detector,
            note_extractor=self._note_extractor,
            feature_extractor=self._feature_extractor,
        )
        self._key_profile_merger: KeyProfileMerger = KeyProfileMerger(logger)
        self._key_profile_transposer: KeyProfileTransposer = KeyProfileTransposer(logger)
        self._num_workers: int = num_workers

        self._processed: int = 0
        self._total: int = 0

        self._normalized_to = template_profile_normalized_to
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def generate_profile(self, corpus_file_path: Path) -> None:
        self._logger.debug(
            f"Generating profiles using corpus: \"{corpus_file_path}\"",
            separator=self._separator,
        )

        corpus = self._read_corpus(corpus_file_path)
        key_profiles = self._profile_all_tracks(corpus)

        self._logger.info(
            f"Successfully generated {len(key_profiles)} profiles out of {len(corpus)}.",
            separator=self._separator,
        )

        merged_profiles = self._key_profile_merger.merge(key_profiles)
        shifted_profiles = self._key_profile_transposer.transpose_profiles_to_c(merged_profiles)
        self._print_profiles(shifted_profiles)

    @staticmethod
    def _read_corpus(path: Path) -> pd.DataFrame:
        """
        Loads exactly the two required columns: 'track_path' and 'track_key'.
        Raises if missing.
        """
        df = pd.read_csv(
            path,
            usecols=["Path", "Principal Key"],
            header=0
        )
        df = df.rename(columns={"Path": "track_path", "Principal Key": "track_key"})
        return df

    def _profile_all_tracks(self, corpus: pd.DataFrame) -> List[KeyProfile]:
        """
        Uses a ThreadPoolExecutor to profile every track in 'corpus'. Returns
        a list of successfully built KeyProfiles.
        """
        total = len(corpus)

        self._processed = 0
        self._total = total

        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            future_to_meta = self._submit_track_jobs(executor, corpus)
            key_profiles = self._collect_track_results(future_to_meta)

        return key_profiles

    def _submit_track_jobs(
            self,
            executor: ThreadPoolExecutor,
            corpus: pd.DataFrame
    ) -> dict:
        """
        Submits each (track_path, track_key) to the thread pool.
        Returns a mapping: Future -> (raw_path, track_key, idx).
        """
        future_to_meta = {}
        for idx, row in enumerate(corpus.itertuples(index=False), start=1):
            raw_path = Path(row.track_path)  # type: ignore
            track_key = row.track_key  # type: ignore

            if not raw_path.exists():
                self._logger.warning(
                    f"Track \"{track_key}\" not found on disk (skipping).",
                    separator=self._separator,
                )
                continue

            pct = idx / self._total * 100
            self._logger.debug(
                f"Scheduling track ({idx}/{self._total} [{pct:.4f}%]): \"{raw_path}\"",
                separator=self._separator,
            )

            future = executor.submit(self._track_profiler.profile, raw_path, track_key)
            future_to_meta[future] = (raw_path, track_key)

        return future_to_meta

    def _collect_track_results(
            self,
            future_to_meta: dict
    ) -> List[KeyProfile]:
        """
        Waits for each Future in future_to_meta to complete. On success,
        collects the KeyProfile; on failure, logs an error.
        """
        collected: List[KeyProfile] = []
        for future in as_completed(future_to_meta):
            self._processed += 1
            raw_path, track_key = future_to_meta[future]
            try:
                kp: KeyProfile = future.result()
                collected.append(kp)
                pct = self._processed / self._total * 100
                self._logger.debug(
                    f"Completed track ({self._processed}/{self._total} [{pct:.4f}%]): "
                    f"\"{raw_path}\" → {kp.get_label()}",
                    separator=self._separator,
                )
            except Exception as exc:
                self._logger.error(
                    f"Error profiling \"{raw_path}\" (label={track_key}): {exc}",
                    separator=self._separator,
                )
        return collected

    def _print_profiles(self, shifted_profiles: List[KeyProfile]) -> None:
        for profile in shifted_profiles:
            mode_vector = self._get_vector_for_mode(profile)
            label = profile.get_label()

            self._logger.debug(f"Generated vector for {label} based on {len(profile.vectors)} tracks.", separator=self._separator)

            self._logger.info(
                f"Template Profile (normalized/precise) for {label}:\n"
                f"{self._format_vector(mode_vector, decimals=18)}",
                separator=self._separator,
            )
            self._logger.info(
                f"Template Profile (normalized/rounded) for {label}:\n"
                f"{self._format_vector(mode_vector, decimals=2)}",
                separator=self._separator,
            )

            self._logger.info("-------------------------------------------------------------", separator=self._separator)

    def _get_vector_for_mode(self, profile: KeyProfile) -> np.ndarray:
        median_profile: np.ndarray = profile.geometric_median()
        self._logger.info(
            f"Template Profile (raw) for {profile.get_label()}:\n{pprint.pformat(median_profile)}",
            separator=self._separator,
        )

        median_profile_sum = median_profile.sum()
        if median_profile_sum == 0:
            self._logger.warning(
                f"Median profile for {profile.get_label()} summed to zero; returning zeros.",
                separator=self._separator,
            )
            return np.zeros_like(median_profile)

        scaling_factor = self._normalized_to / median_profile_sum
        scaled_profile = median_profile * scaling_factor
        return scaled_profile

    @staticmethod
    def _format_vector(vec: np.ndarray, decimals: int = 4) -> str:
        fmt = f"{{:.{decimals}f}}"
        return "[" + ", ".join(fmt.format(x) for x in vec) + "]"
