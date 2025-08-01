import pickle
from pathlib import Path
from typing import Dict, Tuple

from track_analysis.components.track_analysis.features.audio_file_handler import AudioStreamsInfoModel


class MaxRateCache:
    def __init__(self, cache_path: Path):
        self._cache_path = cache_path
        self._cache = self._load()

    def get(self, info: AudioStreamsInfoModel) -> float:
        key = (info.sample_rate_Hz or 0, info.bit_depth or 0, info.channels or 0)
        if 0 in key:
            return 0.0
        if key not in self._cache:
            self._cache[key] = key[0] * key[1] * key[2]
        return self._cache[key]

    def save(self) -> None:
        with open(self._cache_path, 'wb') as f:
            pickle.dump(self._cache, f)

    def _load(self) -> Dict[Tuple[int,int,int], float]:
        if not self._cache_path.exists():
            return {}
        try:
            with open(self._cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return {}
