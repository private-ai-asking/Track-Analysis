import gc
import uuid
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature import AudioDataFeature
from track_analysis.components.track_analysis.features.audio_calculation.audio_data_feature_calculator_orchestrator import \
    AudioDataFeatureCalculatorOrchestrator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.loudness.crest_factor import \
    CrestFactorCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.loudness.integrated_lufs import \
    IntegratedLufsCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.loudness.loudness_analyzer import \
    LoudnessAnalyzer
from track_analysis.components.track_analysis.features.audio_calculation.calculators.loudness.loudness_range import \
    LoudnessRangeCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.loudness.true_peak import \
    TruePeakCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.misc.hpss import HPSSExtractor
from track_analysis.components.track_analysis.features.audio_calculation.calculators.rms.iqr_rms_calculator import \
    IQRRmsCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.rms.max_rms_calculator import \
    MaxRmsCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.rms.mean_rms_calculator import \
    MeanRmsCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.rms.percentile_rms_calculator import \
    PercentileRmsCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.centroid import \
    SpectralCentroidCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.centroid_and_flux import \
    SpectralCentroidAndFluxCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.contrast import \
    SpectralContrastCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.dynamic_tempo import \
    DynamicTempoCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.flatness import \
    SpectralFlatnessCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.flux import \
    SpectralFluxCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.harmonicity import \
    HarmonicityCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.magnitude import \
    HarmonicSpectrogramCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.mfcc import MfccCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.multi_band_onset import \
    MultiBandOnsetPeaksCalculator, MultiBandOnsetEnvelopeCalculator, OnsetEnvMeanKickCalculator, \
    OnsetEnvMeanSnareCalculator, OnsetEnvMeanLowMidCalculator, OnsetEnvMeanHiHatCalculator, OnsetRateKickCalculator, \
    OnsetRateSnareCalculator, OnsetRateLowMidCalculator, OnsetRateHiHatCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.onset_env_mean import \
    OnsetEnvMeanCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.onset_envelope import \
    OnsetEnvelopeCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.onset_peaks import \
    OnsetPeaksCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.onset_rate import \
    OnsetRateCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.rolloff import \
    SpectralRolloffCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.tempo_variation import \
    TempoVariationCalculator
from track_analysis.components.track_analysis.features.audio_calculation.calculators.spectral.zcr import ZCRCalculator
from track_analysis.components.track_analysis.features.audio_calculation.feature_to_header_mapping import \
    FEATURE_TO_HEADER_MAPPING
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler, AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.core.cacheing.magnitude_spectogram import \
    MagnitudeSpectrogramExtractor
from track_analysis.components.track_analysis.features.core.cacheing.multi_band_onset import OnsetStrengthMultiExtractor
from track_analysis.components.track_analysis.features.data_generation.model.album_cost import AlbumCostModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.data_generation.util.key_extractor import KeyExtractor, \
    KeyExtractionResult
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor


class BatchProcessNewTracks(IPipe):
    """
    Processes new audio files by batching file-based and sample-based calculations
    separately for optimal performance and memory management.
    """

    def __init__(
            self,
            logger: HoornLogger,
            file_handler: AudioFileHandler,
            tag_extractor: TagExtractor,
            key_extractor: KeyExtractor,
    ):
        self._logger = logger
        self._file_handler = file_handler
        self._tag_extractor = tag_extractor
        self._key_extractor = key_extractor
        self._separator = "BuildCSV.BatchProcessNewTracks"

        _hop_length = 512
        _n_fft = 2048

        magnitude_extractor = MagnitudeSpectrogramExtractor(logger, n_fft=_n_fft, hop_length=_hop_length)
        onset_multi_extractor = OnsetStrengthMultiExtractor(logger, magnitude_extractor)

        all_calculators = [
            CrestFactorCalculator(), IntegratedLufsCalculator(), LoudnessAnalyzer(),
            LoudnessRangeCalculator(), TruePeakCalculator(), HPSSExtractor(self._logger, hop_length=_hop_length, n_fft=_n_fft),
            IQRRmsCalculator(), MaxRmsCalculator(), MeanRmsCalculator(), PercentileRmsCalculator(),
            SpectralCentroidCalculator(), SpectralCentroidAndFluxCalculator(),
            SpectralContrastCalculator(self._logger, hop_length=_hop_length),
            SpectralFlatnessCalculator(self._logger, hop_length=_hop_length), SpectralFluxCalculator(),
            HarmonicityCalculator(self._logger), MfccCalculator(self._logger),
            MultiBandOnsetPeaksCalculator(onset_multi_extractor), MultiBandOnsetEnvelopeCalculator(onset_multi_extractor),
            OnsetEnvMeanCalculator(), OnsetEnvMeanKickCalculator(), OnsetEnvMeanSnareCalculator(),
            OnsetEnvMeanLowMidCalculator(), OnsetEnvMeanHiHatCalculator(), OnsetRateCalculator(),
            OnsetRateKickCalculator(), OnsetRateSnareCalculator(), OnsetRateLowMidCalculator(),
            OnsetRateHiHatCalculator(), SpectralRolloffCalculator(self._logger, hop_length=_hop_length),
            TempoVariationCalculator(), ZCRCalculator(self._logger, hop_length=_hop_length),
            HarmonicSpectrogramCalculator(self._logger, hop_length=_hop_length, n_fft=_n_fft),
            OnsetEnvelopeCalculator(self._logger, hop_length=_hop_length),
            OnsetPeaksCalculator(self._logger, hop_length=_hop_length),
            DynamicTempoCalculator(self._logger, hop_length=_hop_length),
        ]
        self._orchestrator = AudioDataFeatureCalculatorOrchestrator(all_calculators)

        self._feature_to_header_map = FEATURE_TO_HEADER_MAPPING
        self._features_to_calculate = list(self._feature_to_header_map.keys())

        self._logger.trace("Initialized batch processor.", separator=self._separator)

    def flow(
            self, context: LibraryDataGenerationPipelineContext
    ) -> LibraryDataGenerationPipelineContext:
        """The main execution flow for this pipeline step."""
        paths = context.filtered_audio_file_paths
        batch_size = context.max_new_tracks_per_run

        if not paths:
            self._logger.debug("No new tracks to process.", separator=self._separator)
            self._set_empty_context(context)
            return context

        self._logger.info(
            f"Processing {len(paths)} new tracks in batches of {batch_size}.",
            separator=self._separator,
        )

        all_results_df, all_key_prog_dfs = self._process_all_batches(paths, batch_size, context.album_costs)
        self._map_results_to_context(all_results_df, all_key_prog_dfs, context)

        if not context.generated_audio_info.empty:
            self._logger.info("Applying final energy level calculation...", separator=self._separator)
            context.generated_audio_info = context.energy_calculator.calculate_ratings_for_df(
                context.generated_audio_info, Header.Energy_Level
            )

        self._logger.info("Completed batch processing of new tracks.", separator=self._separator)
        return context

    def _process_all_batches(
            self, paths: List[Path], batch_size: int, album_costs: List[AlbumCostModel]
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """Manages the iteration over all batches and concatenates their results."""
        all_batch_results = []
        all_key_progression_dfs = []
        total = len(paths)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_paths = paths[start:end]

            self._logger.info(
                f"Processing batch {start//batch_size + 1}: Tracks {start+1} to {end} of {total}",
                separator=self._separator
            )

            batch_df, key_prog_dfs = self._process_single_batch(batch_paths, album_costs)
            all_batch_results.append(batch_df)
            all_key_progression_dfs.extend(key_prog_dfs)

        final_df = pd.concat(all_batch_results, ignore_index=True) if all_batch_results else pd.DataFrame()
        return final_df, all_key_progression_dfs

    def _process_single_batch(
            self, batch_paths: List[Path], album_costs: List[AlbumCostModel]
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """Handles a single batch of tracks."""
        stream_infos = self._file_handler.get_audio_streams_info_batch(batch_paths)
        meta_df = self._build_metadata_df(batch_paths, stream_infos, album_costs)

        key_df, key_prog_dfs = self._run_batch_key_extraction(meta_df)
        meta_with_keys_df = meta_df.merge(key_df, left_index=True, right_on='original_index', how='left').drop(columns=['original_index'])

        sample_features = self._run_batch_sample_calculation(meta_with_keys_df, stream_infos)
        calculated_df = pd.DataFrame(sample_features)

        del stream_infos
        gc.collect()

        return pd.concat([meta_with_keys_df.reset_index(drop=True), calculated_df.reset_index(drop=True)], axis=1), key_prog_dfs

    def _run_batch_key_extraction(
            self, meta_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """Runs the specialized KeyExtractor on a batch of file paths."""
        self._logger.info("Starting batch key extraction...", separator=self._separator)
        indexed_paths = list(meta_df[[Header.Audio_Path.value]].itertuples(name=None))
        key_extraction_results = self._key_extractor.extract_keys_batch(indexed_paths)
        return self._build_key_dataframes(key_extraction_results, meta_df)

    def _run_batch_sample_calculation(
            self, meta_df: pd.DataFrame, stream_infos: List[AudioStreamsInfoModel]
    ) -> List[Dict]:
        """Runs the orchestrator for sample-based features for each track in the batch."""
        all_track_features = []
        for _, row in meta_df.iterrows():
            info = next((i for i in stream_infos if i.path == row[Header.Audio_Path.value]), None)
            if not info: continue

            initial_data = {
                AudioDataFeature.AUDIO_SAMPLES: info.samples, AudioDataFeature.AUDIO_SAMPLE_RATE: info.sample_rate_Hz,
                AudioDataFeature.BPM: info.tempo, AudioDataFeature.AUDIO_PATH: info.path,
            }
            calculated_features = self._orchestrator.process_track(initial_data, self._features_to_calculate)
            string_keyed_features = {k.name if isinstance(k, Enum) else k: v for k, v in calculated_features.items()}
            all_track_features.append(string_keyed_features)
        return all_track_features

    @staticmethod
    def _build_key_dataframes(
            key_results: List[KeyExtractionResult], meta_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """Processes raw KeyExtractionResult into a main key df and a list of key progression dfs."""
        main_key_records, key_progression_dfs = [], []
        for res in key_results:
            uuid = meta_df.iloc[res.index][Header.UUID.value]

            # --- THIS IS THE FIX ---
            # Use the correct, final header names when creating the DataFrame.
            progression_df_for_track = pd.DataFrame([
                {
                    "Track UUID": uuid,
                    "Segment Start": lk.interval_start,
                    "Segment End": lk.interval_end,
                    "Segment Key": lk.key,
                } for lk in res.local_info
            ])
            key_progression_dfs.append(progression_df_for_track)

            main_key_records.append({
                'original_index': res.index,
                Header.Key.value: res.global_key,
                Header.Start_Key.value: res.local_info[0].key if res.local_info else None,
                Header.End_Key.value: res.local_info[-1].key if res.local_info else None,
            })
        return pd.DataFrame(main_key_records), key_progression_dfs

    def _map_results_to_context(
            self, full_df: pd.DataFrame, key_prog_dfs: List[pd.DataFrame], context: LibraryDataGenerationPipelineContext
    ) -> None:
        """Maps the combined results DataFrame to the separate context DataFrames."""
        if full_df.empty:
            return self._set_empty_context(context)

        # --- Prepare Main DataFrame ---
        rename_map = {feat.name: head.value for feat, head in self._feature_to_header_map.items() if feat.name in full_df.columns}
        main_df = full_df.rename(columns=rename_map)
        context.generated_audio_info = main_df[[h.value for h in Header if h.value in main_df.columns]]

        # --- Prepare MFCC DataFrame ---
        mfcc_means, mfcc_stds = AudioDataFeature.MFCC_MEANS.name, AudioDataFeature.MFCC_STDS.name
        if mfcc_means in full_df.columns and mfcc_stds in full_df.columns:
            mfcc_data = {Header.UUID.value: full_df[Header.UUID.value]}
            mfcc_means_stacked, mfcc_stds_stacked = np.vstack(full_df[mfcc_means]), np.vstack(full_df[mfcc_stds])
            for i in range(mfcc_means_stacked.shape[1]):
                mfcc_data[f"mfcc_mean_{i}"] = mfcc_means_stacked[:, i]
                mfcc_data[f"mfcc_std_{i}"] = mfcc_stds_stacked[:, i]
            context.generated_mfcc_audio_info = pd.DataFrame(mfcc_data)

        # --- Prepare Key Progression DataFrame ---
        context.generated_key_progression_audio_info = pd.concat(key_prog_dfs, ignore_index=True) if key_prog_dfs else pd.DataFrame()

    def _build_metadata_df(self, paths: List[Path], infos: List[AudioStreamsInfoModel], album_costs: List[AlbumCostModel]) -> pd.DataFrame:
        """Creates a DataFrame with file paths, UUIDs, and extracted tags."""
        records = []
        for path, info in zip(paths, infos):
            record = {
                Header.Audio_Path.value: path,
                Header.UUID.value: str(uuid.uuid4()),
                Header.Duration.value: info.duration,
                Header.Bitrate.value: info.bitrate,
                Header.Bit_Depth.value: info.bit_depth,
                Header.Max_Data_Per_Second.value: info.max_data_per_second_kbps,
                Header.Actual_Data_Rate.value: info.actual_data_rate_kbps,
                Header.Efficiency.value: info.efficiency,
                Header.Sample_Rate.value: info.sample_rate_kHz,
            }
            self._tag_extractor.add_extracted_metadata_to_track(record, info, album_costs)
            records.append(record)
        return pd.DataFrame(records)

    @staticmethod
    def _set_empty_context(context: LibraryDataGenerationPipelineContext) -> None:
        """Helper to set empty DataFrames on the context."""
        context.generated_audio_info = pd.DataFrame()
        context.generated_mfcc_audio_info = pd.DataFrame()
        context.generated_key_progression_audio_info = pd.DataFrame()
