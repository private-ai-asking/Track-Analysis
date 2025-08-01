from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from track_analysis.components.track_analysis.features.audio_calculation.calculators.metric_calculator import \
    AudioMetricCalculator
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class BatchSampleMetricsService:
    def __init__(
            self,
            calculators: List[AudioMetricCalculator],
            num_workers: int
    ):
        self._calculators = calculators
        self._num_workers = num_workers

    def compute(
            self,
            audio_paths: List[Path],
            samples_list: List[np.ndarray],
            sample_rates: List[int],
            tempos: List[float],
            chunk_size: int = 4096
    ) -> dict:
        def _worker(path, samples, sr, tempo):
            result = {}
            for calc in self._calculators:
                result.update(calc.calculate(
                    path,
                    samples,
                    sr,
                    tempo=tempo,
                    chunk_size=chunk_size
                ))
            return result

        with ThreadPoolExecutor(max_workers=self._num_workers) as exe:
            raw = list(exe.map(
                lambda args: _worker(*args),
                zip(audio_paths, samples_list, sample_rates, tempos)
            ))

        out = {}
        for metric_name in raw[0].keys():
            out[metric_name] = np.array([r[metric_name] for r in raw], dtype=np.float32)
        return out

    def build_dataframe(self, sample_metrics: dict) -> pd.DataFrame:
        data = {
            Header.True_Peak.value:                  sample_metrics["true_peak_dbtp"],
            Header.Integrated_LUFS.value:            sample_metrics["integrated_lufs"],
            Header.Program_Dynamic_Range_LRA.value:  sample_metrics["loudness_range_lu"],
            Header.Crest_Factor.value:               sample_metrics["crest_factor_db"],
            Header.Mean_RMS.value:                   sample_metrics["mean_dbfs"],
            Header.Max_RMS.value:                    sample_metrics["max_dbfs"],
            Header.Percentile_90_RMS.value:          sample_metrics["percentile_90_dbfs"],
            Header.RMS_IQR.value:                    sample_metrics["iqr_dbfs"],
            Header.Spectral_Centroid_Mean.value:     sample_metrics["spec_centroid_mean_hz"],
            Header.Spectral_Centroid_Max.value:      sample_metrics["spec_centroid_max_hz"],
            Header.Spectral_Flux_Mean.value:         sample_metrics["spec_flux_mean"],
            Header.Spectral_Flux_Max.value:          sample_metrics["spec_flux_max"],

            # Onsets
            Header.Onset_Env_Mean.value:             sample_metrics["onset_env_mean"],
            Header.Onset_Rate.value:                 sample_metrics["onset_rate"],

            Header.Onset_Env_Mean_Kick.value:             sample_metrics["onset_env_mean_kick"],
            Header.Onset_Rate_Kick.value:                 sample_metrics["onset_rate_kick"],

            Header.Onset_Env_Mean_Snare.value:             sample_metrics["onset_env_mean_snare"],
            Header.Onset_Rate_Snare.value:                 sample_metrics["onset_rate_snare"],

            Header.Onset_Env_Mean_Low_Mid.value:             sample_metrics["onset_env_mean_low_mid"],
            Header.Onset_Rate_Low_Mid.value:                 sample_metrics["onset_rate_low_mid"],

            Header.Onset_Env_Mean_Hi_Hat.value:             sample_metrics["onset_env_mean_hihat"],
            Header.Onset_Rate_Hi_Hat.value:                 sample_metrics["onset_rate_hihat"],
        }
        return pd.DataFrame(data)
