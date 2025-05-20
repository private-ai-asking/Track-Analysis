import os
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List

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

    def _process_row(
            self,
            item: Tuple[int, str],
            stream_info: AudioStreamsInfoModel,
            total: int
    ) -> dict:
        idx, path = item

        # 1. Load and compute
        dyn_range, crest = self._audio_calculator.calculate_program_dynamic_range_and_crest_factor(
            stream_info.samples, int(stream_info.sample_rate_Hz)
        )
        max_dps = self._audio_calculator.calculate_max_data_per_second(stream_info)
        lufs = self._audio_calculator.calculate_lufs(
            stream_info.sample_rate_Hz, stream_info.samples
        )
        true_peak = self._audio_calculator.calculate_true_peak(
            stream_info.sample_rate_Hz, stream_info.samples
        )

        # 2. Compute derived size/rate/efficiency
        size_b = os.path.getsize(path)
        size_bits = size_b * 8
        actual_rate = size_bits / stream_info.duration if stream_info.duration > 0 else 0
        efficiency = (actual_rate / max_dps) * 100 if max_dps > 0 else 0

        self._processed += 1

        self._logger.trace(f"Finished adding metadata for track: {path}",
                           separator=self._separator)
        self._logger.info(
            f"Processed {self._processed}/{total} ({self._processed/total*100:.2f}%) tracks.",
            separator=self._separator,
        )

        return {
            "idx": idx,
            Header.Duration.value: stream_info.duration,
            Header.Bitrate.value: stream_info.bitrate,
            Header.Sample_Rate.value: stream_info.sample_rate_kHz,
            Header.Peak_To_RMS.value: dyn_range,
            Header.Crest_Factor.value: crest,
            Header.Bit_Depth.value: stream_info.bit_depth,
            Header.Max_Data_Per_Second.value: max_dps / 1000,
            Header.Actual_Data_Rate.value: actual_rate / 1000,
            Header.Efficiency.value: efficiency,
            Header.Format.value: stream_info.format,
            Header.Loudness.value: lufs,
            Header.True_Peak.value: true_peak,
        }

    def flow(
            self,
            data: LibraryDataGenerationPipelineContext
    ) -> LibraryDataGenerationPipelineContext:
        df = data.generated_audio_info
        stream_infos: List[AudioStreamsInfoModel] = data.extracted_stream_info
        total = len(df)
        self._logger.trace("Adding advanced metadata...", separator=self._separator)

        if not stream_infos or len(stream_infos) != total:
            raise ValueError(
                f"Expected {total} stream info entries, got {len(stream_infos)}"
            )

        # pair each DataFrame row (idx, path) with its corresponding stream_info
        self._processed = 0
        iterator = zip(df[Header.Audio_Path.value].items(), stream_infos)

        if data.use_threads:
            with ThreadPoolExecutor() as exe:
                results = list(
                    exe.map(lambda pair: self._process_row(pair[0], pair[1], total), iterator)
                )
        else:
            results = [
                self._process_row(item, info, total)
                for item, info in iterator
            ]

        self._audio_calculator.save_cache()

        res_df = pd.DataFrame(results).set_index("idx")
        data.generated_audio_info = df.join(res_df, how="left")

        self._logger.info("Finished adding advanced metadata.", separator=self._separator)
        return data
