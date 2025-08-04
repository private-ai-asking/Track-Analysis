import pandas as pd
from enum import Enum
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_calculator_orchestrator import \
    AudioDataFeatureCalculatorOrchestrator
from track_analysis.components.track_analysis.features.audio_file_handler import AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class SampleFeatureProcessor:
    def __init__(
            self,
            orchestrator: AudioDataFeatureCalculatorOrchestrator,
            features_to_calculate: List[AudioDataFeature],
            logger: HoornLogger,
            num_workers: int = 4,
    ):
        self._orchestrator = orchestrator
        self._features_to_calculate = features_to_calculate
        self._logger = logger
        self._separator = "BuildCSV.SampleFeatureProcessor"
        self._num_workers = num_workers

    def _process_track(self, row_info_tuple: Tuple, stream_infos: List[AudioStreamsInfoModel]) -> Dict[str, Any] | None:
        """Helper function to process a single track; designed to be called by the thread pool."""
        _, row = row_info_tuple
        info = next((i for i in stream_infos if i.path == row[Header.Audio_Path.value]), None)

        if not info:
            self._logger.warning(f"No stream info found for path: {row[Header.Audio_Path.value]}", separator=self._separator)
            return None

        initial_data = {
            AudioDataFeature.AUDIO_SAMPLES: info.samples,
            AudioDataFeature.AUDIO_SAMPLE_RATE: info.sample_rate_Hz,
            AudioDataFeature.BPM: info.tempo,
            AudioDataFeature.AUDIO_PATH: info.path,
        }
        calculated_features = self._orchestrator.process_track(initial_data, self._features_to_calculate)
        return {k.name if isinstance(k, Enum) else k: v for k, v in calculated_features.items()}

    def process_batch(
            self, meta_df: pd.DataFrame, stream_infos: List[AudioStreamsInfoModel]
    ) -> pd.DataFrame:
        """Runs the orchestrator for sample-based features for each track in the batch using multithreading."""
        self._logger.info(f"Starting batch sample calculation with {self._num_workers} workers...", separator=self._separator)
        all_track_features = []

        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            future_to_row = {executor.submit(self._process_track, row_info, stream_infos): row_info for row_info in meta_df.iterrows()}

            for future in as_completed(future_to_row):
                result = future.result()
                if result:
                    all_track_features.append(result)

        return pd.DataFrame(all_track_features)
