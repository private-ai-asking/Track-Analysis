import h5py
import numpy as np
import pickle
import json
import time
import hashlib
from pathlib import Path
from threading import Lock
from typing import Any, Union, Dict

class HDF5CacheManager:
    """
    Thread-safe, SWMR-compatible hierarchical HDF5 cache.

    This cache uses a hierarchical structure to store data, organized as:
    `/{track_id}/{feature}/{params_hash}`

    This allows for extremely fast lookups without loading the entire cache
    into memory.
    """
    def __init__(self, cache_path: Union[str, Path]):
        self._cache_path = Path(cache_path)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock() # Lock is still needed to protect write operations

        # Open one long-lived writer handle and enable SWMR
        # 'a' mode is sufficient as we will create groups and datasets.
        self._file = h5py.File(self._cache_path, 'a', libver='latest')
        self._file.swmr_mode = True

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitizes a string to be a valid HDF5 group/dataset name."""
        # HDF5 names are UTF-8 but cannot contain '/' or be '.' or '..'
        return name.replace('/', '_')

    @staticmethod
    def _get_params_hash(params: Dict[str, Any]) -> str:
        """Creates a stable SHA256 hash of a parameters dictionary."""
        if not params:
            return 'no_params'
        p_json = json.dumps(params, sort_keys=True).encode('utf-8')
        return hashlib.sha256(p_json).hexdigest()

    def _get_item_path(self, track_id: str, feature: str, params: Dict[str, Any]) -> str:
        """Constructs the HDF5 internal path for a given cache item."""
        sanitized_track = self._sanitize_name(track_id)
        sanitized_feature = self._sanitize_name(feature)
        params_hash = self._get_params_hash(params)
        return f"/{sanitized_track}/{sanitized_feature}/{params_hash}"

    def put(self,
            track_id:    str,
            feature:     str,
            params:      Dict[str,Any],
            data:        Any):
        """Saves data to the cache if it does not already exist."""
        item_path = self._get_item_path(track_id, feature, params)

        with self._lock:
            if item_path in self._file:
                return

            # Create parent groups if they don't exist.
            group_path = str(Path(item_path).parent)
            self._file.require_group(group_path)

            # Serialize the data
            payload = pickle.dumps(data)

            # Create the dataset and store the payload
            dset = self._file.create_dataset(item_path, data=np.frombuffer(payload, dtype='uint8'))

            # Store metadata as attributes on the dataset
            dset.attrs['timestamp'] = time.time()
            dset.attrs['params_json'] = json.dumps(params, sort_keys=True)

            self._file.flush()

    def get(self,
            track_id:  str,
            feature:   str,
            params:    Dict[str,Any]) -> Any | None:
        """Retrieves an item from the cache. Returns None if not found."""
        item_path = self._get_item_path(track_id, feature, params)

        # Use a new read-only, SWMR-enabled handle for thread-safety
        try:
            with h5py.File(self._cache_path, 'r', swmr=True, libver='latest') as f:
                if item_path not in f:
                    return None

                payload_bytes = f[item_path][()].tobytes()
                return pickle.loads(payload_bytes)
        except (OSError, FileNotFoundError):
            return None


    def exists(self,
               track_id: str,
               feature:  str,
               params:   Dict[str,Any]) -> bool:
        """
        Quickly checks if a matching item exists in the cache.
        This is a very fast metadata-only operation.
        """
        item_path = self._get_item_path(track_id, feature, params)
        try:
            with h5py.File(self._cache_path, 'r', swmr=True, libver='latest') as f:
                return item_path in f
        except (OSError, FileNotFoundError):
            return False

    def close(self):
        """Closes the main file handle."""
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None

    def __del__(self):
        self.close()
