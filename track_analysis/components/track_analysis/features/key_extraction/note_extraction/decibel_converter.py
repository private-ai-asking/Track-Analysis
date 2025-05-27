import numpy as np
from librosa import amplitude_to_db

class DecibelConverter:
    @staticmethod
    def to_db(chroma: np.ndarray) -> np.ndarray:
        return amplitude_to_db(chroma, ref=np.max)
