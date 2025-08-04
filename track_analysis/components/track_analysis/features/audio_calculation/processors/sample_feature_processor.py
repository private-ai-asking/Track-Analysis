from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import List, Dict, Any, Tuple

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_provider_orchestrator import \
    AudioDataFeatureProviderOrchestrator
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class SampleFeatureProcessor:
    def __init__(
            self,
            orchestrator: AudioDataFeatureProviderOrchestrator,
            features_to_calculate: List[AudioDataFeature],
            logger: HoornLogger,
            num_workers: int = 4,
    ):
        self._orchestrator = orchestrator
        self._features_to_calculate = features_to_calculate
        self._logger = logger
        self._separator = "BuildCSV.SampleFeatureProcessor"
        self._num_workers = num_workers

        self._processed: int = 0
        self._to_process: int = 0

    def _process_track(self, row_info_tuple: Tuple) -> Dict[str, Any] | None:
        """Helper function to process a single track; designed to be called by the thread pool."""
        _, row = row_info_tuple

        initial_data = {
            AudioDataFeature.AUDIO_PATH: row[Header.Audio_Path.value],
        }

        calculated_features = self._orchestrator.process_track(initial_data, self._features_to_calculate)

        self._processed += 1
        self._logger.info(f"Processed {self._processed} / {self._to_process} ({self._processed/self._to_process*100:.2f}%) tracks.", separator=self._separator)

        return {k.name if isinstance(k, Enum) else k: v for k, v in calculated_features.items()}

    def process_batch(
            self, meta_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Runs the orchestrator for sample-based features for each track in the batch using multithreading."""
        self._logger.info(f"Starting batch sample calculation with {self._num_workers} workers...", separator=self._separator)
        all_track_features = []

        self._processed = 0
        self._to_process = len(meta_df)

        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            future_to_row = {executor.submit(self._process_track, row_info): row_info for row_info in meta_df.iterrows()}

            for future in as_completed(future_to_row):
                result = future.result()
                if result:
                    all_track_features.append(result)

        return pd.DataFrame(all_track_features)
