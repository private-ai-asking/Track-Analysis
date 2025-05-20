import os
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List

import numpy as np
import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.audio_calculator import AudioCalculator
from track_analysis.components.track_analysis.features.audio_file_handler import AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext


class AddAdvancedMetadata(IPipe):
    def __init__(
            self,
            logger: HoornLogger,
            audio_calculator: AudioCalculator,
    ):
        self._separator = "BuildCSV.AddAdvancedMetadataPipe"
        self._logger = logger
        self._audio_calculator = audio_calculator
        self._processed: int = 0
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: LibraryDataGenerationPipelineContext):
        df   = data.generated_audio_info
        infos = data.extracted_stream_info

        # 1) batch crest/peak
        peaks, rmss = self._audio_calculator.calculate_batch_crest(
            [info.samples for info in infos]
        )
        crest_dbs = 20.0 * np.log10(peaks / rmss)

        # 2) batch the rest
        rest = self._audio_calculator.calculate_batch_rest(infos, df[Header.Audio_Path.value].tolist())

        # 3) assemble one result DataFrame
        result = {
            "idx":         df.index.values,
            Header.Crest_Factor.value: crest_dbs,
            Header.Peak_To_RMS.value:  np.array([self._audio_calculator.calculate_program_dr(i.samples, i.sample_rate_Hz) for i in infos]),
            **rest
        }
        res_df = pd.DataFrame(result).set_index("idx")

        # 4) join in one shot
        data.generated_audio_info = df.join(res_df, how="left")
        return data
