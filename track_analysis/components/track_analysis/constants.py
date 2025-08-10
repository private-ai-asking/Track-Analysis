import os
from enum import Enum, auto
from pathlib import Path
from typing import Dict

import psutil

from track_analysis.components.track_analysis.library.audio_transformation.feature_extraction.audio_data_feature import \
    AudioDataFeature

ROOT: Path = Path(os.path.realpath(__file__)).parent.parent.parent.parent

CONFIGURATION_PATH: Path = ROOT / "track_analysis_configuration.json"
PHYSICAL_CPU_COUNT: int = psutil.cpu_count(logical=False)

# ======================================================================================================================

class MFCCType(Enum):
    MEANS = auto()
    STDS = auto()
    VELOCITY_MEANS = auto()
    VELOCITY_STDS = auto()
    ACCELERATION_MEANS = auto()
    ACCELERATION_STDS = auto()

# Calculated - leave alone.
MFFC_LABEL_PREFIXES: Dict[MFCCType, str] = {
    MFCCType.MEANS: "mfcc_mean_",
    MFCCType.STDS: "mfcc_std_",
    MFCCType.VELOCITY_MEANS: "velocity_mean_",
    MFCCType.VELOCITY_STDS: "velocity_std_",
    MFCCType.ACCELERATION_MEANS: "acceleration_mean_",
    MFCCType.ACCELERATION_STDS: "acceleration_std_"
}

_MFCC_TYPE_TO_AUDIO_DATA_FEATURE: Dict[MFCCType, AudioDataFeature] = {
    MFCCType.MEANS: AudioDataFeature.MFCC_MEANS,
    MFCCType.STDS: AudioDataFeature.MFCC_STDS,
    MFCCType.VELOCITY_MEANS: AudioDataFeature.MFCC_VELOCITIES_MEANS,
    MFCCType.VELOCITY_STDS: AudioDataFeature.MFCC_VELOCITIES_STDS,
    MFCCType.ACCELERATION_MEANS: AudioDataFeature.MFCC_ACCELERATIONS_MEANS,
    MFCCType.ACCELERATION_STDS: AudioDataFeature.MFCC_ACCELERATIONS_STDS,
}

MFCC_LABEL_PREFIX_TO_AUDIO_DATA_FEATURE: Dict[str, AudioDataFeature] = {
    MFFC_LABEL_PREFIXES[prefix]: feature
    for prefix, feature in _MFCC_TYPE_TO_AUDIO_DATA_FEATURE.items()
}
