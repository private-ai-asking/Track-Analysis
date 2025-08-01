from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from track_analysis.components.track_analysis.features.audio_calculation.batch_file_metrics_service import \
    BatchFileMetricsService
from track_analysis.components.track_analysis.features.audio_calculation.batch_sample_metrics_service import \
    BatchSampleMetricsService
from track_analysis.components.track_analysis.features.audio_file_handler import AudioStreamsInfoModel


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
            samples_list: List[np.ndarray],
            sample_rates: List[int],
            tempos: List[float],
    ) -> pd.DataFrame:
        sample_metrics = self._sample_service.compute(
            paths, samples_list, sample_rates, tempos
        )
        df_samples = self._sample_service.build_dataframe(sample_metrics)

        df_files = self._file_service.compute(infos, paths)

        return pd.concat([df_samples, df_files], axis=1)
