from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf


class AudioSampleLoader:
    """Loads audio samples into memory."""
    @staticmethod
    def load_audio_samples(audio_path: Path) -> Tuple[np.ndarray, float]:
        """Returns the samples and the sample rate."""
        samples, sr = sf.read(audio_path, dtype='float32')
        return samples, sr
