import math
import os
import pickle
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import mutagen
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pyebur128.pyebur128 import R128State, MeasurementMode, get_loudness_range, get_loudness_global, get_true_peak

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import EXPENSIVE_CACHE_DIRECTORY
from track_analysis.components.track_analysis.features.audio_file_handler import AudioStreamsInfoModel
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.util.key_extractor import KeyExtractor

def compute_short_time_rms_dbfs(
        samples: np.ndarray,
        sr: int,
        window_ms: float = 50.0,
        hop_ms: float = 10.0
) -> Tuple[float, float, float]:
    """
    Compute short-time RMS energy statistics in dBFS.

    Returns:
        mean_db  : Mean RMS level in dBFS
        max_db   : Maximum RMS level in dBFS
        p90_db   : 90th-percentile RMS level in dBFS
    """
    # 1) Ensure mono
    if samples.ndim == 2:
        mono = samples.mean(axis=1)
    else:
        mono = samples

    # 2) Convert time-based window/hop sizes to samples
    frame_length = int(sr * window_ms / 1000)
    hop_length   = int(sr * hop_ms    / 1000)
    if frame_length < 1:
        raise ValueError(f"window_ms {window_ms}ms too small for sample rate {sr}")

    # 3) Fallback for short signals
    total_frames = mono.shape[0]
    if total_frames < frame_length:
        whole_rms = np.sqrt(np.mean(mono**2))
        db = 20.0 * math.log10(max(whole_rms, np.finfo(float).eps))
        return db, db, db

    # 4) Frame the signal
    n_frames = 1 + (total_frames - frame_length) // hop_length
    trimmed = mono[: hop_length * n_frames + frame_length - hop_length]
    windows = sliding_window_view(trimmed, frame_length)[::hop_length]

    # 5) Compute RMS per window
    rms_vals = np.sqrt(np.mean(windows**2, axis=1))
    rms_vals = np.maximum(rms_vals, np.finfo(rms_vals.dtype).eps)

    # 6) Convert to dBFS and compute stats
    rms_dbfs = 20.0 * np.log10(rms_vals)
    mean_db  = float(rms_dbfs.mean())
    max_db   = float(rms_dbfs.max())
    p90_db   = float(np.percentile(rms_dbfs, 90))
    return mean_db, max_db, p90_db


class AudioCalculator:
    def __init__(self, logger: HoornLogger, key_progression_path: Path, num_workers: int):
        self._separator = "AudioCalculator"
        self._logger = logger

        self._key_progression_path: Path = key_progression_path
        self._key_extractor: KeyExtractor = KeyExtractor(logger, num_workers)

        self._max_data_rate_cache_path: Path = EXPENSIVE_CACHE_DIRECTORY / "max_data_rate_cache.pkl"
        self._max_data_rate_cache_lookup: Dict[Tuple[int, int, int], float] = self._load_max_data_rate_cache()

        self._processed = 0
        self._logger.trace("Successfully initialized.", separator=self._separator)

    # -------------------- Public API --------------------

    def calculate_batch_sample_metrics(
            self,
            samples_list: List[np.ndarray],
            sample_rates: List[int],
            chunk_size: int = 4096,
            max_workers: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate True-Peak, Integrated LUFS, Loudness Range, and Crest Factor
        for each sample in samples_list, in parallel.
        """
        self._processed = 0

        results = self._run_sample_metric_workers(samples_list, sample_rates, chunk_size, max_workers)
        tps, lufs, lras, crest_db, rms_data = zip(*results)
        mean_rms, max_rms, p90_rms = zip(*rms_data)

        return {
            Header.True_Peak.value:                  np.array(tps,       dtype=np.float32),
            Header.Integrated_LUFS.value:            np.array(lufs,      dtype=np.float32),
            Header.Program_Dynamic_Range_LRA.value:  np.array(lras,      dtype=np.float32),
            Header.Crest_Factor.value:               np.array(crest_db,  dtype=np.float32),
            Header.Mean_RMS.value:                   np.array(mean_rms,  dtype=np.float32),
            Header.Max_RMS.value:                    np.array(max_rms,   dtype=np.float32),
            Header.Percentile_90_RMS.value:          np.array(p90_rms,   dtype=np.float32),
        }

    def calculate_batch_rest(
            self,
            infos: List[AudioStreamsInfoModel],
            paths: List[Path]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate duration, bitrate, sample rate, data rates, efficiency, bit depth,
        plus extract principal and segment keys for each track.
        """
        durations, bitrates, sample_rates, maxdps, bit_depths = self._compute_scalar_arrays(infos)
        actual_bits = self._compute_actual_data_rates(paths, infos)
        efficiencies = self._compute_efficiency(actual_bits, maxdps)

        extractor_results = self._extract_keys(paths)
        global_keys, start_keys, end_keys, segment_rows = self._process_extractor_results(extractor_results, paths)
        self._append_segments_to_csv(segment_rows)

        return {
            Header.Duration.value:             durations,
            Header.Bitrate.value:              bitrates,
            Header.Sample_Rate.value:          sample_rates,
            Header.Max_Data_Per_Second.value:  maxdps / 1_000,
            Header.Actual_Data_Rate.value:     actual_bits / 1_000,
            Header.Efficiency.value:           efficiencies,
            Header.Bit_Depth.value:            bit_depths,
            Header.Key.value:                  global_keys,
            Header.Start_Key.value:            start_keys,
            Header.End_Key.value:              end_keys
        }

    def save_cache(self) -> None:
        """
        Persist the max-data-rate cache to disk.
        """
        try:
            with open(self._max_data_rate_cache_path, "wb") as f:
                pickle.dump(self._max_data_rate_cache_lookup, f)
        except Exception as e:
            self._logger.error(f"Failed to save cache: {e}", separator=self._separator)

    # -------------------- Sample Metrics Helpers --------------------

    def _run_sample_metric_workers(
            self,
            samples_list: List[np.ndarray],
            sample_rates: List[int],
            chunk_size: int,
            max_workers: Optional[int]
    ) -> List[Tuple[float, float, float, float, Tuple[float, float, float]]]:
        """
        Spawn threads to compute metrics for each (samples, sr) pair.
        """
        total = len(samples_list)
        self._processed = 0

        def _worker(samples: np.ndarray, sr: int) -> Tuple[float, float, float, float, Tuple[float, float, float]]:
            true_peak, lufs, lra, crest_db, rms_data = self._compute_loudness_metrics(samples, sr, chunk_size)
            self._processed += 1
            self._logger.info(
                f"Processed {self._processed}/{total} ({self._processed/total*100:.2f}%) tracks.",
                separator=self._separator
            )
            return true_peak, lufs, lra, crest_db, rms_data

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_worker, samples_list, sample_rates))

    # noinspection t
    @staticmethod
    def _compute_loudness_metrics(
            samples: np.ndarray,
            sr: int,
            chunk_size: int
    ) -> Tuple[float, float, float, float, Tuple[float, float, float]]:
        """
        Compute True-Peak (dBTP), Integrated LUFS, Loudness Range (LU),
        Crest Factor (dB), and short-time RMS stats (dBFS) for one sample.
        """
        # --- existing loudness and crest computation ---
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]
        frames, channels = samples.shape

        st_i  = R128State(channels, sr, MeasurementMode.MODE_I)
        st_lra= R128State(channels, sr, MeasurementMode.MODE_LRA)
        st_tp = R128State(channels, sr, MeasurementMode.MODE_TRUE_PEAK)

        sum_sq = 0.0
        max_abs_sq = 0.0
        for offset in range(0, frames, chunk_size):
            block = samples[offset: offset + chunk_size]
            interleaved = block.flatten()
            n = block.shape[0]
            st_i.add_frames(interleaved, n)
            st_lra.add_frames(interleaved, n)
            st_tp.add_frames(interleaved, n)
            sq = block ** 2
            sum_sq += sq.sum()
            max_abs_sq = max(max_abs_sq, sq.max())

        lufs     = get_loudness_global(st_i)
        lra      = get_loudness_range(st_lra)
        tp_ch    = [20 * math.log10(get_true_peak(st_tp, ch)) for ch in range(channels)]
        true_peak= max(tp_ch) if tp_ch else 0.0

        peak     = math.sqrt(max_abs_sq)
        rms_all  = math.sqrt(sum_sq / (frames * channels)) if frames > 0 else 0.0
        crest_db = 20.0 * math.log10(peak / rms_all) if rms_all > 0 else 0.0

        mean_db, max_db, p90_db = compute_short_time_rms_dbfs(
            samples if channels == 1 else samples.mean(axis=1, keepdims=False),
            sr
        )

        return true_peak, lufs, lra, crest_db, (mean_db, max_db, p90_db)

    # -------------------- Batch Rest Helpers --------------------

    def _compute_scalar_arrays(
            self,
            infos: List[AudioStreamsInfoModel]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return arrays for durations, bitrates, sample rates, max data/sec, and bit depths.
        """
        durations = np.array([i.duration for i in infos], dtype=np.float32)
        bitrates = np.array([i.bitrate for i in infos], dtype=np.float32)
        sample_rates = np.array([i.sample_rate_kHz for i in infos], dtype=np.float32)
        maxdps = np.array(
            [self._calculate_max_data_per_second(i) for i in infos],
            dtype=np.float32
        )
        bit_depths = np.array([i.bit_depth for i in infos], dtype=np.float32)
        return durations, bitrates, sample_rates, maxdps, bit_depths

    @staticmethod
    def _compute_actual_data_rates(
            paths: List[Path],
            infos: List[AudioStreamsInfoModel]
    ) -> np.ndarray:
        """
        Return array of actual bits/sec for each file in paths.
        """
        actual_bits = np.empty(len(paths), dtype=np.float32)
        for idx, (p, info) in enumerate(zip(paths, infos)):
            size_b = os.path.getsize(p)
            size_bits = size_b * 8
            actual_bits[idx] = size_bits / info.duration if info.duration > 0 else 0.0
        return actual_bits

    @staticmethod
    def _compute_efficiency(
            actual_bits: np.ndarray,
            maxdps: np.ndarray
    ) -> np.ndarray:
        """
        Return efficiency percentage array based on actual bits/sec and max data/sec.
        """
        return np.where(maxdps > 0, actual_bits / maxdps * 100.0, 0.0)

    def _extract_keys(
            self,
            paths: List[Path]
    ) -> List[Tuple[int, str, List[Tuple[float, float, str]]]]:
        """
        Call KeyExtractor on the list of paths, returning (idx, global_key, segments).
        """
        indexed_paths = [(idx, path) for idx, path in enumerate(paths)]
        try:
            return self._key_extractor.extract_keys_batch(indexed_paths)
        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error(f"KeyExtractor failed: {e}\n{tb}", separator=self._separator)
            return []

    def _process_extractor_results(
            self,
            extractor_results: List[Tuple[int, str, List[Tuple[float, float, str]]]],
            paths: List[Path]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]]]:
        """
        From extractor_results, build arrays for global, start, end keys, and a list of segment rows.
        """
        count = len(paths)
        global_keys = np.full(count, "", dtype=object)
        start_keys = np.full(count, "", dtype=object)
        end_keys = np.full(count, "", dtype=object)

        segment_rows: List[Dict[str, object]] = []

        for idx, raw_global_key, segments in extractor_results:
            global_keys[idx] = raw_global_key
            track_path = paths[idx]

            # Write principal key to file if .flac
            if track_path.suffix.lower() == ".flac":
                self._write_flac_key_tag(track_path, raw_global_key)

            if segments:
                first_label = segments[0][2]
                last_label = segments[-1][2]
                start_keys[idx] = first_label
                end_keys[idx] = last_label

                for start_t, end_t, raw_label in segments:
                    segment_rows.append({
                        "File Path":     str(track_path),
                        "Segment Start": start_t,
                        "Segment End":   end_t,
                        "Segment Key":   raw_label
                    })

        return global_keys, start_keys, end_keys, segment_rows

    def _write_flac_key_tag(self, path: Path, key: str) -> None:
        """
        Write 'initialkey' and 'global_key' tags into a .flac file.
        """
        try:
            tag_file = mutagen.File(str(path), easy=True)
            tag_file["initialkey"] = key
            tag_file["global_key"] = key
            tag_file.save()
        except Exception:
            # Log if desired, but do not interrupt flow
            pass

    def _append_segments_to_csv(self, segment_rows: List[Dict[str, object]]) -> None:
        """
        Append segment_rows to the key progression CSV. Create file if it doesn't exist.
        """
        if not segment_rows:
            return

        try:
            df_segments = pd.DataFrame(segment_rows)
            df_segments.to_csv(
                self._key_progression_path,
                mode="a",
                index=False,
                header=not self._key_progression_path.exists()
            )
        except Exception as e:
            self._logger.error(f"Failed to append segments CSV: {e}", separator=self._separator)

    # -------------------- Max Data Rate Cache --------------------

    def _calculate_max_data_per_second(self, info: AudioStreamsInfoModel) -> float:
        """
        Calculate or look up bits-per-second for a given stream_info.
        """
        depth = info.bit_depth or 0
        sr = info.sample_rate_Hz or 0
        channels = info.channels or 0

        if depth == 0 or sr == 0 or channels == 0:
            return 0.0

        key = (sr, depth, channels)
        cached = self._max_data_rate_cache_lookup.get(key)
        if cached is not None:
            return cached

        rate = sr * depth * channels
        self._max_data_rate_cache_lookup[key] = rate
        return rate

    def _load_max_data_rate_cache(self) -> Dict[Tuple[int, int, int], float]:
        """
        Load the max-data-rate cache from disk, or return empty dict if not found.
        """
        if not self._max_data_rate_cache_path.exists():
            self._logger.warning("Max data rate cache not found; initializing empty.", separator=self._separator)
            return {}
        try:
            with open(self._max_data_rate_cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            self._logger.error(f"Failed to load data rate cache: {e}", separator=self._separator)
            return {}
