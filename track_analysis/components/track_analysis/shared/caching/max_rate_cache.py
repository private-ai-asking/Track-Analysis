import pickle
from pathlib import Path
from typing import Dict, Tuple


class MaxRateCache:
    def __init__(self, cache_path: Path):
        self._cache_path = cache_path
        self._cache = self._load()

    def compute_max_rate_bps(self, sample_rate: int, bit_depth: int, channels: int) -> float:
        key = self._get_key(sample_rate, bit_depth, channels)
        if 0 in key:
            return 0.0

        return sample_rate * bit_depth * channels

    def get(self, sample_rate_hz: int | None, bit_depth: int | None, channels: int | None) -> Tuple[float, bool]:
        key = self._get_key(sample_rate_hz, bit_depth, channels)

        if key not in self._cache:
            return 0.0, False

        return self._cache[key], True

    def put(self, sample_rate_hz: int, bit_depth: int, channels: int, value: float) -> None:
        key = self._get_key(sample_rate_hz, bit_depth, channels)

        if key not in self._cache:
            self._cache[key] = value

    @staticmethod
    def _get_key(sample_rate_hz: int | None, bit_depth: int | None, channels: int | None) -> Tuple[int, int, int]:
        return sample_rate_hz or 0, bit_depth or 0, channels or 0

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
