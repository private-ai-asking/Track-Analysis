import dataclasses
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from track_analysis.components.track_analysis.features.audio_calculation.batch_file_metrics_service import \
    BatchFileMetricsService
from track_analysis.components.track_analysis.features.audio_calculation.batch_sample_metrics_service import \
    BatchSampleMetricsService
from track_analysis.components.track_analysis.features.audio_file_handler import AudioStreamsInfoModel

@dataclasses.dataclass(frozen=True)
class AudioProcessingResult:
    main_df: pd.DataFrame
    mfcc_df: pd.DataFrame
    key_progression_df: pd.DataFrame


class AudioProcessingPipeline:
    def __init__(
            self,
            sample_service: BatchSampleMetricsService,
            file_service:   BatchFileMetricsService
    ):
        self._sample_service = sample_service
        self._file_service   = file_service

    def run(
            self,
            infos: List[AudioStreamsInfoModel],
            paths: List[Path],
            uuids: List[str],
            samples_list: List[np.ndarray],
            sample_rates: List[int],
            tempos: List[float],
    ) -> AudioProcessingResult:
        # Pass UUIDs to the sample service
        sample_metrics = self._sample_service.compute(
            uuids, paths, samples_list, sample_rates, tempos
        )

        # Build both DataFrames
        df_main_samples = self._sample_service.build_main_dataframe(sample_metrics)
        df_mfcc = self._sample_service.build_mfcc_dataframe(sample_metrics)

        df_files, df_key_progression = self._file_service.compute(infos, paths, uuids)
        df_main = pd.concat([df_main_samples, df_files], axis=1)

        return AudioProcessingResult(df_main, df_mfcc, df_key_progression)
