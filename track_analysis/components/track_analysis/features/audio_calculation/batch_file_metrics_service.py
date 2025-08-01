from typing import List
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
            paths: List[Path]
    ) -> pd.DataFrame:
        count = len(paths)
        df = pd.DataFrame({
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

        df[Header.Actual_Data_Rate.value]    = np.array(actual, dtype=np.float32) / 1e3
        df[Header.Max_Data_Per_Second.value] = np.array(maxdps,   dtype=np.float32) / 1e3
        df[Header.Efficiency.value]          = np.where(
            df[Header.Max_Data_Per_Second.value] > 0,
            df[Header.Actual_Data_Rate.value] / df[Header.Max_Data_Per_Second.value] * 100,
            0.0
        )

        # key-tagging + note-rate
        gk, sk, ek, results = self._key_tagger.tag_and_record(paths)
        df[Header.Key.value]       = gk
        df[Header.Start_Key.value] = sk
        df[Header.End_Key.value]   = ek

        # now build note_rate from the returned results
        note_rate = np.zeros(count, dtype=np.float32)
        for res in results:
            dur = infos[res.index].duration or 1.0
            note_rate[res.index] = len(res.note_events) / dur
        df[Header.Onset_Rate_Notes.value] = note_rate

        return df
