import os

import librosa
import numpy as np
from joblib import Memory

from track_analysis.components.track_analysis.constants import CACHE_DIRECTORY


def _convert_to_db(audio: np.ndarray, ref=np.max) -> np.ndarray:
    return librosa.amplitude_to_db(audio, ref=ref)

cache_dir = CACHE_DIRECTORY / "convert to db"
os.makedirs(cache_dir, exist_ok=True)
convert_to_db_func = Memory(cache_dir, verbose=0).cache(_convert_to_db)
