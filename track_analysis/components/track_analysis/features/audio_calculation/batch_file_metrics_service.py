from typing import List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

from track_analysis.components.track_analysis.features.audio_calculation.key_tagging_service import KeyTaggingService
from track_analysis.components.track_analysis.features.audio_calculation.utils.cacheing.max_rate_cache import \
    MaxRateCache
from track_analysis.components.track_analysis.features.audio_calculation.utils.file_utils import FileUtils
from track_analysis.components.track_analysis.features.audio_file_handler import AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header


class BatchFileMetricsService:
    def __init__(
            self,
            file_utils: FileUtils,
            rate_cache: MaxRateCache,
            key_tagger: KeyTaggingService
    ):
        self._file_utils = file_utils
        self._rate_cache = rate_cache
        self._key_tagger = key_tagger

    def compute(
            self,
            infos: List[AudioStreamsInfoModel],
            paths: List[Path],
            uuids: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        count = len(paths)
        main_df = pd.DataFrame({
            Header.UUID.value: uuids,

            Header.Duration.value:    [i.duration        for i in infos],
            Header.Bitrate.value:     [i.bitrate         for i in infos],
            Header.Sample_Rate.value: [i.sample_rate_kHz for i in infos],
            Header.Bit_Depth.value:   [i.bit_depth       for i in infos],
        })

        # data-rate
        actual = [
            (self._file_utils.get_size_bytes(p) * 8) / i.duration
            if i.duration > 0 else 0.0
            for p, i in zip(paths, infos)
        ]
        maxdps = [ self._rate_cache.get(i) for i in infos ]

        main_df[Header.Actual_Data_Rate.value]    = np.array(actual, dtype=np.float32) / 1e3
        main_df[Header.Max_Data_Per_Second.value] = np.array(maxdps,   dtype=np.float32) / 1e3
        main_df[Header.Efficiency.value]          = np.where(
            main_df[Header.Max_Data_Per_Second.value] > 0,
            main_df[Header.Actual_Data_Rate.value] / main_df[Header.Max_Data_Per_Second.value] * 100,
            0.0
        )

        # key-tagging + note-rate
        results = self._key_tagger.tag_and_record(paths)
        main_df[Header.Key.value]       = results.global_keys
        main_df[Header.Start_Key.value] = results.start_keys
        main_df[Header.End_Key.value]   = results.end_keys

        # now build note_rate from the returned results
        note_rate = np.zeros(count, dtype=np.float32)
        for key_extraction_result in results.key_extraction_results:
            dur = infos[key_extraction_result.index].duration or 1.0
            note_rate[key_extraction_result.index] = len(key_extraction_result.note_events) / dur
        main_df[Header.Onset_Rate_Notes.value] = note_rate

        return main_df, results.key_progression_df
