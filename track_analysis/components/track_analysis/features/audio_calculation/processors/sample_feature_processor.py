from enum import Enum
from typing import List

import pandas as pd

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
    ):
        self._orchestrator = orchestrator
        self._features_to_calculate = features_to_calculate
        self._logger = logger
        self._separator = "BuildCSV.SampleFeatureProcessor"

    def process_batch(
            self, meta_df: pd.DataFrame, stream_infos: List[AudioStreamsInfoModel]
    ) -> pd.DataFrame:
        """Runs the orchestrator for sample-based features for each track in the batch."""
        self._logger.info("Starting batch sample calculation...", separator=self._separator)
        all_track_features = []
        for _, row in meta_df.iterrows():
            info = next((i for i in stream_infos if i.path == row[Header.Audio_Path.value]), None)
            if not info:
                self._logger.warning(f"No stream info found for path: {row[Header.Audio_Path.value]}", separator=self._separator)
                continue

            initial_data = {
                AudioDataFeature.AUDIO_SAMPLES: info.samples,
                AudioDataFeature.AUDIO_SAMPLE_RATE: info.sample_rate_Hz,
                AudioDataFeature.BPM: info.tempo,
                AudioDataFeature.AUDIO_PATH: info.path,
            }
            calculated_features = self._orchestrator.process_track(initial_data, self._features_to_calculate)
            string_keyed_features = {k.name if isinstance(k, Enum) else k: v for k, v in calculated_features.items()}
            all_track_features.append(string_keyed_features)

        return pd.DataFrame(all_track_features)
