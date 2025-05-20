import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.audio_calculator import AudioCalculator
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler, AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext


class AddAdvancedMetadata(IPipe):
    def __init__(
            self,
            logger: HoornLogger,
            audio_file_handler: AudioFileHandler,
            audio_calculator: AudioCalculator,
    ):
        self._separator = "BuildCSV.AddAdvancedMetadataPipe"
        self._logger = logger
        self._audio_file_handler = audio_file_handler
        self._audio_calculator = audio_calculator
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def _process_row(self, item: pd.Series):
        idx, path = item

        # 1. Load and compute
        file_info: AudioStreamsInfoModel = self._audio_file_handler.get_audio_streams_info(path)
        dyn_range, crest = self._audio_calculator.calculate_dynamic_range_and_crest_factor(
            file_info.samples_librosa
        )
        max_dps = self._audio_calculator.calculate_max_data_per_second(file_info)
        lufs = self._audio_calculator.calculate_lufs(
            file_info.sample_rate_Hz, file_info.samples_librosa
        )
        true_peak = self._audio_calculator.calculate_true_peak(
            file_info.sample_rate_Hz, file_info.samples_librosa
        )

        # 2. Compute derived size/rate/efficiency
        size_b = os.path.getsize(path)
        size_bits = size_b * 8
        actual_rate = size_bits / file_info.duration if file_info.duration > 0 else 0
        efficiency = (actual_rate / max_dps) * 100 if max_dps > 0 else 0

        return {
            "idx": idx,
            Header.Duration.value: file_info.duration,
            Header.Bitrate.value: file_info.bitrate,
            Header.Sample_Rate.value: file_info.sample_rate_kHz,
            Header.Peak_To_RMS.value: dyn_range,
            Header.Crest_Factor.value: crest,
            Header.Bit_Depth.value: file_info.bit_depth,
            Header.Max_Data_Per_Second.value: max_dps / 1000,
            Header.Actual_Data_Rate.value: actual_rate / 1000,
            Header.Efficiency.value: efficiency,
            Header.Format.value: file_info.format,
            Header.Loudness.value: lufs,
            Header.True_Peak.value: true_peak,
        }

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        df = data.generated_audio_info
        total = len(df)
        self._logger.trace("Adding advanced metadata...", separator=self._separator)

        # If you expect heavy I/O you can swap Sequential vs ThreadPool
        iterator = df[Header.Audio_Path.value].items()
        if data.use_threads:
            with ThreadPoolExecutor() as exe:
                results = list(exe.map(self._process_row, iterator))
        else:
            results = [self._process_row(item) for item in iterator]

        # 3. Assign back into DataFrame, logging each step
        for count, res in enumerate(results, start=1):
            idx = res.pop("idx")
            for col, val in res.items():
                df.loc[idx, col] = val    # recommended bracket + .loc pattern :contentReference[oaicite:1]{index=1}
            path = df.loc[idx, Header.Audio_Path.value]
            self._logger.trace(f"Finished adding metadata for track: {path}", separator=self._separator)
            self._logger.info(
                f"Processed {count}/{total} ({count/total*100:.4f}%) tracks.",
                separator=self._separator,
            )

        self._logger.info("Finished adding advanced metadata.", separator=self._separator)
        return data
