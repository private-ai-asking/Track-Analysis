import traceback
from collections import defaultdict
from dataclasses import dataclass

import h5py
import numpy as np
import pickle
import json
import time
import hashlib
from pathlib import Path
from threading import Lock
from typing import Any, Union, Dict, List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger

@dataclass
class CacheEntryInfo:
    """A simple data structure to hold information about a single cache entry."""
    path: str
    size: int


class HDF5CacheManager:
    """
    Thread-safe HDF5 cache that opens on creation and closes on deletion.
    Designed for a simple, global decorator-based workflow like joblib.Memory.
    """
    def __init__(self, cache_path: Union[str, Path]):
        self._cache_path = Path(cache_path)
        self._lock = Lock()
        self._logger: HoornLogger | None = None
        self._separator: str = self.__class__.__name__

        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = h5py.File(self._cache_path, 'a', libver='latest')
            self._file.swmr_mode = True
        except Exception as e:
            raise RuntimeError(f"Failed to create or open HDF5 file at '{self._cache_path}': {e}") from e

    def set_logger(self, logger: HoornLogger):
        self._logger = logger
        if self._logger:
            self._logger.debug(f"Logger set. Cache manager is active for file: '{self._cache_path}'", separator=self._separator)

    # --- Public API Methods ---

    def put(self, track_id: str, feature: str, params: Dict[str, Any], data: Any):
        """Saves data atomically to the cache if it does not already exist."""
        item_path = self._get_item_path(track_id, feature, params)
        if self._logger:
            self._logger.trace(f"Attempting to cache item: '{item_path}'", separator=self._separator)

        with self._lock:
            if item_path in self._file:
                if self._logger:
                    self._logger.trace(f"Item already in cache, skipping put: '{item_path}'", separator=self._separator)
                return

            self._write_item_to_file(item_path, params, data)

    def get(self, track_id: str, feature: str, params: Dict[str, Any]) -> Any | None:
        """Retrieves an item from the cache. Returns None if not found."""
        item_path = self._get_item_path(track_id, feature, params)
        if self._logger:
            self._logger.trace(f"Attempting to retrieve item: '{item_path}'", separator=self._separator)

        payload_bytes = self._read_payload_from_file(item_path)
        if payload_bytes is None:
            return None # Miss or I/O error already logged.

        data = self._deserialize_payload(payload_bytes, item_path)
        if data is not None and self._logger:
            self._logger.debug(f"Cache hit: '{item_path}'", separator=self._separator)

        return data

    def close(self):
        """Closes the main file handle."""
        with self._lock:
            if self._file:
                if self._logger:
                    self._logger.debug(f"Closing HDF5 cache file: '{self._cache_path}'", separator=self._separator)
                self._file.close()
                self._file = None

    def __del__(self):
        """Ensures the file is closed when the object is garbage collected."""
        if self._logger and self._file:
            self._logger.debug(f"Closing cache file via __del__ (garbage collection): '{self._cache_path}'", separator=self._separator)
        self.close()

    # --- Private Helper Methods ---

    def _write_item_to_file(self, item_path: str, params: Dict[str, Any], data: Any):
        """Serializes and writes a single item to the HDF5 file atomically."""
        try:
            # Step 1: Prepare all data and metadata before touching the file.
            payload = pickle.dumps(data)
            params_json = json.dumps(params, sort_keys=True)

            # Decompose the path into its parent and the final dataset name.
            path_obj = Path(item_path)
            parent_group_path = str(path_obj.parent)
            dataset_name = path_obj.name

            # Step 2: Get a handle to the parent group, creating it if it doesn't exist.
            parent_group = self._file.require_group(parent_group_path)

            # Step 3: Create the new dataset RELATIVE to the parent group handle.
            dset = parent_group.create_dataset(dataset_name, data=np.frombuffer(payload, dtype='uint8'))

            # Step 4: Set attributes and flush.
            dset.attrs['timestamp'] = time.time()
            dset.attrs['params_json'] = params_json
            self._file.flush()

            if self._logger:
                self._logger.debug(f"Cached new item: '{item_path}'", separator=self._separator)

        except Exception as e:
            if self._logger:
                tb = traceback.format_exc()
                self._logger.error(f"Failed to write item '{item_path}' to cache. Error: {e}\n{tb}", separator=self._separator)
            return

    def _read_payload_from_file(self, item_path: str) -> bytes | None:
        """
        Reads the raw byte payload for an item from the HDF5 file.
        This method manages the file handle and its I/O errors.
        """
        try:
            with h5py.File(self._cache_path, 'r', swmr=True, libver='latest') as f:
                return self._get_dataset_bytes(f, item_path)
        except (OSError, FileNotFoundError) as e:
            if self._logger:
                self._logger.error(f"Failed to open or read cache file '{self._cache_path}': {e}", separator=self._separator)
            return None
        except Exception as e:
            if self._logger:
                self._logger.error(f"An unexpected error occurred reading dataset for '{item_path}': {e}", separator=self._separator)
            return None

    def _get_dataset_bytes(self, hdf5_file: h5py.File, item_path: str) -> bytes | None:
        """
        Gets the bytes of a dataset from an open HDF5 file, logging a cache miss if not found.
        """
        if item_path not in hdf5_file:
            if self._logger:
                self._logger.debug(f"Cache miss: '{item_path}'", separator=self._separator)
            return None
        return hdf5_file[item_path][()].tobytes()

    def _deserialize_payload(self, payload_bytes: bytes, item_path: str) -> Any | None:
        """Converts a raw byte payload into a Python object."""
        try:
            return pickle.loads(payload_bytes)
        except Exception as e:
            if self._logger:
                tb = traceback.format_exc()
                self._logger.error(f"Failed to deserialize data from '{item_path}'. Cache entry may be corrupt. Error: {e}\n{tb}", separator=self._separator)
            return None

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return name.replace('/', '_')

    @staticmethod
    def _get_params_hash(params: Dict[str, Any]) -> str:
        if not params:
            return 'no_params'
        p_json = json.dumps(params, sort_keys=True).encode('utf-8')
        return hashlib.sha256(p_json).hexdigest()

    def _get_item_path(self, track_id: str, feature: str, params: Dict[str, Any]) -> str:
        sanitized_track = self._sanitize_name(track_id)
        sanitized_feature = self._sanitize_name(feature)
        params_hash = self._get_params_hash(params)
        return f"/{sanitized_track}/{sanitized_feature}/{params_hash}"

    def log_statistics(self, top_n: int = 5):
        """
        Calculates and logs detailed statistics about the cache file.

        Args:
            top_n (int): The number of top space-consuming items to list.
        """
        if not self._logger:
            print("Warning: Logger not set. Cannot log statistics.")
            return

        with self._lock:
            if not self._file:
                self._logger.warning("Cache file is not open. Cannot log statistics.", separator=self._separator)
                return

            try:
                entry_data = self._collect_entry_data()

                if not entry_data:
                    report = f"HDF5 Cache Statistics for '{self._cache_path}'\n" + "-"*40 + "\n  Cache is empty."
                    self._logger.info(report, separator=self._separator)
                    return

                stats = self._calculate_statistics(entry_data, top_n)
                report = self._format_statistics_report(stats)

                self._logger.info(report, separator=self._separator)

            except Exception as e:
                tb = traceback.format_exc()
                self._logger.error(f"Failed to generate cache statistics. Error: {e}\n{tb}", separator=self._separator)

    def _collect_entry_data(self) -> List[CacheEntryInfo]:
        """Traverses the HDF5 file and collects info about each dataset."""
        entries = []
        def visitor_func(name, obj):
            if isinstance(obj, h5py.Dataset):
                size_bytes = obj.size * obj.dtype.itemsize
                entries.append(CacheEntryInfo(path=name, size=size_bytes))

        self._file.visititems(visitor_func)
        return entries

    def _calculate_statistics(self, entry_data: List[CacheEntryInfo], top_n: int) -> Dict[str, Any]:
        """Calculates aggregate statistics from the collected entry data."""
        stats = {}
        all_sizes = [entry.size for entry in entry_data]

        # Basic size stats
        stats['num_entries'] = len(all_sizes)
        stats['total_data_size'] = sum(all_sizes)
        stats['avg_size'] = np.mean(all_sizes)
        stats['std_dev_size'] = np.std(all_sizes)
        stats['min_size'] = min(all_sizes) if all_sizes else 0
        stats['max_size'] = max(all_sizes) if all_sizes else 0

        # Track-based stats
        track_sizes = defaultdict(int)
        for entry in entry_data:
            track_id = entry.path.split('/')[0]
            track_sizes[track_id] += entry.size

        # Feature-based stats
        feature_sizes = defaultdict(int)
        for entry in entry_data:
            parts = entry.path.split('/')
            if len(parts) >= 2:
                feature_path = f"/{parts[0]}/{parts[1]}"
                feature_sizes[feature_path] += entry.size

        stats['num_unique_tracks'] = len(track_sizes)
        stats['num_unique_features'] = len(feature_sizes)
        stats['top_n'] = top_n
        stats['file_size_bytes'] = self._cache_path.stat().st_size

        # Sort and get top N
        stats['top_tracks'] = sorted(track_sizes.items(), key=lambda item: item[1], reverse=True)[:top_n]
        stats['top_features'] = sorted(feature_sizes.items(), key=lambda item: item[1], reverse=True)[:top_n]

        return stats

    def _format_statistics_report(self, stats: Dict[str, Any]) -> str:
        """Formats the calculated statistics into a readable report string."""
        total_size = stats['total_data_size']

        # noinspection PyListCreation
        report = [f"HDF5 Cache Statistics for '{self._cache_path}'", "="*60]
        report.append(f"  Overall File Size: {self._format_bytes(stats['file_size_bytes'])}")
        report.append(f"  Total Cached Entries: {stats['num_entries']}")
        report.append(f"  Unique Tracks: {stats['num_unique_tracks']}")
        report.append(f"  Unique Feature Types: {stats['num_unique_features']}")
        report.append("-" * 40)

        report.append("  Internal Data Size Distribution:")
        report.append(f"    - Total: {self._format_bytes(total_size)}")
        report.append(f"    - Average: {self._format_bytes(stats['avg_size'])}")
        report.append(f"    - Std Dev: {self._format_bytes(stats['std_dev_size'])}")
        report.append(f"    - Min | Max: {self._format_bytes(stats['min_size'])} | {self._format_bytes(stats['max_size'])}")
        report.append("-" * 40)

        report.append(f"  Top {stats['top_n']} Tracks by Size:")
        for track, size in stats['top_tracks']:
            percentage = (size / total_size) * 100 if total_size > 0 else 0
            report.append(f"    - {track}: {self._format_bytes(size)} ({percentage:.2f}%)")
        report.append("-" * 40)

        report.append(f"  Top {stats['top_n']} Feature Types by Size:")
        for feature, size in stats['top_features']:
            percentage = (size / total_size) * 100 if total_size > 0 else 0
            report.append(f"    - {feature}: {self._format_bytes(size)} ({percentage:.2f}%)")
        report.append("="*60)

        return "\n".join(report)

    @staticmethod
    def _format_bytes(size_bytes: int | float) -> str:
        """Formats a byte size into a human-readable string."""
        if size_bytes is None:
            return "N/A"
        if size_bytes < 1024:
            return f"{size_bytes} B"
        size_kb = size_bytes / 1024
        if size_kb < 1024:
            return f"{size_kb:.2f} KB"
        size_mb = size_kb / 1024
        if size_mb < 1024:
            return f"{size_mb:.2f} MB"
        size_gb = size_mb / 1024
        return f"{size_gb:.2f} GB"
