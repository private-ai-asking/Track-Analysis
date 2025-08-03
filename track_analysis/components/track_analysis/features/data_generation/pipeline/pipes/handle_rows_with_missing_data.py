import gc
import time
from pathlib import Path
from typing import List, Dict, Callable

import numpy as np
import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.md_common_python.py_common.time_handling import TimeUtils
from track_analysis.components.track_analysis.features.audio_calculation.batch_sample_metrics_service import \
    BatchSampleMetricsService
from track_analysis.components.track_analysis.features.audio_calculation.calculators.rms_calculator import \
    compute_short_time_rms_dbfs
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral_rhythm_calculator import \
    SpectralRhythmCalculator, FeatureExtractors
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler, AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.core.cacheing.harmonic import HarmonicExtractor
from track_analysis.components.track_analysis.features.core.cacheing.harmonicity import HarmonicityExtractor
from track_analysis.components.track_analysis.features.core.cacheing.magnitude_spectogram import \
    MagnitudeSpectrogramExtractor
from track_analysis.components.track_analysis.features.core.cacheing.mfcc import MfccExtractor
from track_analysis.components.track_analysis.features.core.cacheing.multi_band_onset import OnsetStrengthMultiExtractor
from track_analysis.components.track_analysis.features.core.cacheing.onset_envelope import OnsetStrengthExtractor
from track_analysis.components.track_analysis.features.core.cacheing.spectral_contrast import SpectralContrastExtractor
from track_analysis.components.track_analysis.features.core.cacheing.spectral_flatness import SpectralFlatnessExtractor
from track_analysis.components.track_analysis.features.core.cacheing.spectral_rolloff import SpectralRolloffExtractor
from track_analysis.components.track_analysis.features.core.cacheing.zero_crossing import ZeroCrossingRateExtractor
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext



class HandleRowsWithMissingData(IPipe):
    def __init__(self, logger: HoornLogger, file_handler: AudioFileHandler):
        self._separator = "BuildCSV.HandleRowsWithMissingData"
        self._file_handler = file_handler

        self._hop_length = 512
        self._onset_extractor = OnsetStrengthExtractor(logger)
        self._harmonic_extractor = HarmonicExtractor(logger)
        self._magnitude_extractor = MagnitudeSpectrogramExtractor(logger)
        onset_extractor_multi = OnsetStrengthMultiExtractor(logger, magnitude_extractor=self._magnitude_extractor)
        zcr_extractor = ZeroCrossingRateExtractor(logger)
        flatness_extractor = SpectralFlatnessExtractor(logger)
        contrast_extractor = SpectralContrastExtractor(logger)
        rolloff = SpectralRolloffExtractor(logger)
        harmonicity = HarmonicityExtractor(logger)
        mfcc = MfccExtractor(logger)

        self._header_processor_func_mapping: Dict[Header, Callable[[List[str], LibraryDataGenerationPipelineContext], None]] = {
            Header.BPM: self._handle_missing_bpm,
            Header.Bit_Depth: self._handle_missing_bit_depth,
            Header.Start_Key: self._handle_missing_segment_keys,
            Header.End_Key: self._handle_missing_segment_keys,
            Header.Mean_RMS: self._handle_missing_rms,
            Header.Max_RMS: self._handle_missing_rms,
            Header.Percentile_90_RMS: self._handle_missing_rms,
            Header.RMS_IQR: self._handle_missing_rms,
            Header.Spectral_Rolloff_Mean: self._handle_missing_spectral
        }

        extractors: FeatureExtractors = FeatureExtractors(
            harmonic=self._harmonic_extractor,
            magnitude=self._magnitude_extractor,
            onset_global=self._onset_extractor,
            onset_multi=onset_extractor_multi,
            zcr=zcr_extractor,
            flatness=flatness_extractor,
            contrast=contrast_extractor,
            rolloff=rolloff,
            harmonicity=harmonicity,
            mfcc=mfcc,
        )
        self._spectral_calculator = SpectralRhythmCalculator(extractors, hop_length=512)
        self._time_utils: TimeUtils = TimeUtils()

        self._logger = logger
        self._logger.trace("Successfully initialized pipe.", separator=self._separator)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        for header, uuids in data.missing_headers.items():
            processor = self._header_processor_func_mapping.get(header)
            if processor is None:
                self._logger.warning(
                    f"Header {header} not found, unsupported refill... skipping.",
                    separator=self._separator
                )
                continue
            processor(uuids, data)

        return data

    def _get_existing_tempos(
            self,
            chunk: pd.DataFrame
    ) -> Dict[Path, float]:
        """Extracts BPM values for each audio path in this chunk."""
        return {
            Path(p): bpm
            for p, bpm in zip(
                chunk[Header.Audio_Path.value],
                chunk[Header.BPM.value]
            )
        }

    def _handle_missing_bit_depth(self, uuids: List[str], data: LibraryDataGenerationPipelineContext) -> None:
        df: pd.DataFrame = data.loaded_audio_info_cache

        # 1) select only rows whose UUID is in uuids **and** whose extension is “.flac”
        rows: pd.DataFrame = df.loc[
            df[Header.UUID.value].isin(uuids) &
            (df[Header.Extension.value].str.lower() == ".flac")
            ]

        self._logger.info(
            f"Found {len(rows)} .flac tracks with missing bit depth.",
            separator=self._separator
        )

        # 2) build Path list (up to your max per run)
        paths: List[Path] = [
                                Path(p) for p in rows[Header.Audio_Path.value].tolist()
                            ][: data.max_new_tracks_per_run]

        # 3) fetch new bit depths
        audio_infos: List[AudioStreamsInfoModel] = self._file_handler.get_audio_streams_info_batch(paths)
        bit_depths: np.ndarray = np.array(
            [i.bit_depth for i in audio_infos],
            dtype=np.float64
        )

        # 4) write back into df at the correct indices
        idxs = rows.index[: len(bit_depths)]
        df.loc[idxs, Header.Bit_Depth.value] = bit_depths

        self._logger.info(
            f"Finished processing {len(paths)} .flac tracks with missing bit depths!",
            separator=self._separator
        )

    def _handle_missing_bpm(self, uuids: List[str], data: LibraryDataGenerationPipelineContext) -> None:
        df: pd.DataFrame = data.loaded_audio_info_cache
        batch_size = data.max_new_tracks_per_run

        mask = df[Header.UUID.value].isin(uuids)
        rows = df.loc[mask]
        total = len(rows)
        self._logger.info(
            f"Found {total} tracks with missing BPM stats.",
            separator=self._separator
        )

        # Process in batches to limit memory usage
        for start in range(0, total, batch_size):
            end = start + batch_size
            rows_chunk = rows.iloc[start:end]
            paths = [Path(p) for p in rows_chunk[Header.Audio_Path.value].tolist()]
            if not paths:
                continue

            # Load audio info and samples for this batch
            stream_infos: List[AudioStreamsInfoModel] = \
                self._file_handler.get_audio_streams_info_batch(paths)

            # Compute RMS stats per track
            bpm = [info.tempo for info in stream_infos]

            # Write back into DataFrame for this chunk
            idxs = rows_chunk.index

            df.loc[idxs, Header.BPM.value] = np.array(bpm, dtype=np.float64)

            self._logger.info(
                f"Filled BPM stats for tracks {start+1} to {min(end, total)}.",
                separator=self._separator
            )

            # Cleanup to free memory
            del stream_infos, bpm
            gc.collect()

    # noinspection t
    def _handle_missing_segment_keys(self, uuids: List[str], data: LibraryDataGenerationPipelineContext) -> None:
        df: pd.DataFrame = data.loaded_audio_info_cache

        # ——— Ensure Start_Key and End_Key exist as object columns ———
        for col in (Header.Start_Key.value, Header.End_Key.value):
            if col not in df.columns:
                df[col] = ""
            else:
                df[col] = df[col].astype(object)

        key_csv_path: Path = Path(data.key_progression_output_file_path)
        if not key_csv_path.exists():
            self._logger.warning(
                f"Key progression CSV not found at {key_csv_path}; cannot fill start/end keys.",
                separator=self._separator
            )
            return

        # Load existing key progression CSV
        segments_df = pd.read_csv(key_csv_path)
        filtered = segments_df.loc[segments_df["Track UUID"].isin(uuids)]

        if filtered.empty:
            self._logger.info(
                "No segments found for given UUIDs; nothing to fill.",
                separator=self._separator
            )
            return

        # Group by UUID to find first and last segments
        grouped = filtered.groupby("Track UUID")

        for uuid, group in grouped:
            # Find row with minimum "Segment Start" for first key
            first_row = group.loc[group["Segment Start"].idxmin()]
            first_key = first_row["Segment Key"]
            # Find row with maximum "Segment End" for last key
            last_row = group.loc[group["Segment End"].idxmax()]
            last_key = last_row["Segment Key"]

            # Find index in main DataFrame for this UUID
            idxs = df.index[df[Header.UUID.value] == uuid].tolist()
            if not idxs:
                continue
            idx = idxs[0]

            # Only fill if missing in df
            if pd.isna(df.loc[idx, Header.Start_Key.value]) or df.loc[idx, Header.Start_Key.value] == "":
                df.loc[idx, Header.Start_Key.value] = first_key
            if pd.isna(df.loc[idx, Header.End_Key.value]) or df.loc[idx, Header.End_Key.value] == "":
                df.loc[idx, Header.End_Key.value] = last_key

        self._logger.info(
            f"Filled start/end keys for {len(grouped)} tracks from key progression CSV.",
            separator=self._separator
        )

    def _handle_missing_rms(self, uuids: List[str], data: LibraryDataGenerationPipelineContext) -> None:
        df: pd.DataFrame = data.loaded_audio_info_cache
        batch_size = data.max_new_tracks_per_run

        # Select rows needing RMS (mean, max or 90th missing)
        mask = df[Header.UUID.value].isin(uuids)
        rows = df.loc[mask]
        total = len(rows)
        self._logger.info(
            f"Found {total} tracks with missing RMS stats.",
            separator=self._separator
        )

        # Process in batches to limit memory usage
        for start in range(0, total, batch_size):
            end = start + batch_size
            rows_chunk = rows.iloc[start:end]
            paths = [Path(p) for p in rows_chunk[Header.Audio_Path.value].tolist()]
            if not paths:
                continue

            # Load audio info and samples for this batch
            stream_infos: List[AudioStreamsInfoModel] = \
                self._file_handler.get_audio_streams_info_batch(paths)

            # Compute RMS stats per track
            stats = [compute_short_time_rms_dbfs(info.samples, info.sample_rate_Hz)
                     for info in stream_infos]
            mean_vals, max_vals, p90_vals, rms_iqr_vals = zip(*stats)

            # Write back into DataFrame for this chunk
            idxs = rows_chunk.index
            df.loc[idxs, Header.Mean_RMS.value] = np.array(mean_vals, dtype=np.float64)
            df.loc[idxs, Header.Max_RMS.value]  = np.array(max_vals,  dtype=np.float64)
            df.loc[idxs, Header.Percentile_90_RMS.value] = np.array(p90_vals, dtype=np.float64)
            df.loc[idxs, Header.RMS_IQR.value] = np.array(rms_iqr_vals, dtype=np.float64)

            self._logger.info(
                f"Filled RMS stats for tracks {start+1} to {min(end, total)}.",
                separator=self._separator
            )

            # Cleanup to free memory
            del stream_infos, stats
            gc.collect()

    def _handle_missing_spectral(self, uuids: List[str], data: LibraryDataGenerationPipelineContext) -> None:
        df_main: pd.DataFrame = data.loaded_audio_info_cache
        df_mfcc: pd.DataFrame = data.loaded_mfcc_info_cache
        batch_size = data.max_new_tracks_per_run

        mask = df_main[Header.UUID.value].isin(uuids)
        rows = df_main.loc[mask]
        total = len(rows)
        self._logger.info(
            f"Found {total} tracks with missing Spectral and MFCC stats.",
            separator=self._separator
        )

        # Process in batches to limit memory usage
        for start in range(0, total, batch_size):
            start_time = time.time()
            end = start + batch_size
            rows_chunk = rows.iloc[start:end]
            paths = [Path(p) for p in rows_chunk[Header.Audio_Path.value].tolist()]
            if not paths:
                continue

            existing_tempos = self._get_existing_tempos(rows_chunk)

            # Load audio info and samples for this batch
            stream_infos: List[AudioStreamsInfoModel] = \
                self._file_handler.get_audio_streams_info_batch(paths, existing_tempos=existing_tempos)

            # Compute all stats per track (including MFCCs)
            stats = [
                self._spectral_calculator.calculate(
                    audio_path=info.path,
                    samples=info.samples,
                    sr=info.sample_rate_Hz,
                    tempo=info.tempo
                )
                for info in stream_infos
            ]

            # --- UNPACK MAIN METRICS ---
            spec_centroid_mean_hz    = [s["spec_centroid_mean_hz"]   for s in stats]
            spec_centroid_max_hz     = [s["spec_centroid_max_hz"]    for s in stats]
            spec_flux_mean           = [s["spec_flux_mean"]          for s in stats]
            spec_flux_max            = [s["spec_flux_max"]           for s in stats]
            zcr_mean                 = [s["zcr_mean"]                for s in stats]
            spectral_flatness_mean   = [s["spectral_flatness_mean"]  for s in stats]
            spectral_contrast_mean   = [s["spectral_contrast_mean"]  for s in stats]
            onset_env_mean           = [s["onset_env_mean"]          for s in stats]
            onset_rate               = [s["onset_rate"]              for s in stats]
            onset_env_mean_kick      = [s["onset_env_mean_kick"]     for s in stats]
            onset_rate_kick          = [s["onset_rate_kick"]         for s in stats]
            onset_env_mean_snare     = [s["onset_env_mean_snare"]    for s in stats]
            onset_rate_snare         = [s["onset_rate_snare"]        for s in stats]
            onset_env_mean_low_mid   = [s["onset_env_mean_low_mid"]  for s in stats]
            onset_rate_low_mid       = [s["onset_rate_low_mid"]      for s in stats]
            onset_env_mean_hihat     = [s["onset_env_mean_hihat"]    for s in stats]
            onset_rate_hihat         = [s["onset_rate_hihat"]        for s in stats]
            spec_rolloff_mean        = [s["spec_rolloff_mean"]       for s in stats]
            spec_rolloff_std         = [s["spec_rolloff_std"]        for s in stats]
            tempo_variation          = [s["tempo_variation"]         for s in stats]
            harmonicity              = [s["harmonicity"]             for s in stats]

            # --- UNPACK MFCC METRICS ---
            mfcc_means               = [s["mffcc_means"]             for s in stats]
            mfcc_stds                = [s["mfcc_stds"]               for s in stats]

            # Write back into df_main (loaded_audio_info_cache)
            idxs = rows_chunk.index
            df_main.loc[idxs, Header.Spectral_Centroid_Mean.value]    = np.array(spec_centroid_mean_hz,   dtype=np.float64)
            df_main.loc[idxs, Header.Spectral_Centroid_Max.value]     = np.array(spec_centroid_max_hz,    dtype=np.float64)
            df_main.loc[idxs, Header.Spectral_Flux_Mean.value]        = np.array(spec_flux_mean,          dtype=np.float64)
            df_main.loc[idxs, Header.Spectral_Flux_Max.value]         = np.array(spec_flux_max,           dtype=np.float64)
            df_main.loc[idxs, Header.Zero_Crossing_Rate_Mean.value]   = np.array(zcr_mean,                dtype=np.float64)
            df_main.loc[idxs, Header.Spectral_Flatness_Mean.value]    = np.array(spectral_flatness_mean,  dtype=np.float64)
            df_main.loc[idxs, Header.Spectral_Contrast_Mean.value]    = np.array(spectral_contrast_mean,  dtype=np.float64)
            df_main.loc[idxs, Header.Spectral_Rolloff_Mean.value]     = np.array(spec_rolloff_mean,       dtype=np.float64)
            df_main.loc[idxs, Header.Spectral_Rolloff_Std.value]      = np.array(spec_rolloff_std,        dtype=np.float64)
            df_main.loc[idxs, Header.Tempo_Variation.value]           = np.array(tempo_variation,         dtype=np.float64)
            df_main.loc[idxs, Header.Harmonicity.value]               = np.array(harmonicity,             dtype=np.float64)
            df_main.loc[idxs, Header.Onset_Env_Mean.value]            = np.array(onset_env_mean,          dtype=np.float64)
            df_main.loc[idxs, Header.Onset_Rate.value]                = np.array(onset_rate,              dtype=np.float64)
            df_main.loc[idxs, Header.Onset_Env_Mean_Kick.value]       = np.array(onset_env_mean_kick,     dtype=np.float64)
            df_main.loc[idxs, Header.Onset_Rate_Kick.value]           = np.array(onset_rate_kick,         dtype=np.float64)
            df_main.loc[idxs, Header.Onset_Env_Mean_Snare.value]      = np.array(onset_env_mean_snare,    dtype=np.float64)
            df_main.loc[idxs, Header.Onset_Rate_Snare.value]          = np.array(onset_rate_snare,        dtype=np.float64)
            df_main.loc[idxs, Header.Onset_Env_Mean_Low_Mid.value]    = np.array(onset_env_mean_low_mid,  dtype=np.float64)
            df_main.loc[idxs, Header.Onset_Rate_Low_Mid.value]        = np.array(onset_rate_low_mid,      dtype=np.float64)
            df_main.loc[idxs, Header.Onset_Env_Mean_Hi_Hat.value]     = np.array(onset_env_mean_hihat,    dtype=np.float64)
            df_main.loc[idxs, Header.Onset_Rate_Hi_Hat.value]         = np.array(onset_rate_hihat,        dtype=np.float64)

            # --- WRITE BACK TO MFCC DATAFRAME  ---
            chunk_uuids = rows_chunk[Header.UUID.value].tolist()

            # Recreate the sample_metrics dict for the chunk
            mfcc_metrics_chunk = {
                Header.UUID.value: chunk_uuids,
                "mffcc_means": mfcc_means,
                "mfcc_stds": mfcc_stds
            }

            # Build the new MFCC DataFrame for this chunk using the corrected static method
            mfcc_df_chunk = BatchSampleMetricsService.build_mfcc_dataframe(mfcc_metrics_chunk)

            # Merge the new chunk into the main MFCC DataFrame.
            # This handles both updates and new rows in a single operation.
            data.loaded_mfcc_info_cache = pd.merge(
                df_mfcc,
                mfcc_df_chunk,
                on=Header.UUID.value,
                how='outer',
                suffixes=('_old', '')
            )

            end_time = time.time()
            elapsed = end_time - start_time
            elapsed_formatted = self._time_utils.format_time(elapsed, round_digits=4)

            self._logger.info(
                f"Filled Spectral and MFCC stats for tracks {start+1} to {min(end, total)} [duration: {elapsed_formatted}].",
                separator=self._separator
            )

            # Cleanup to free memory
            del stream_infos, stats, mfcc_metrics_chunk, mfcc_df_chunk
            gc.collect()
