from enum import Enum, auto
from typing import List


class AudioDataFeature(Enum):
    """
    Enumeration for audio features, organized logically from raw file properties
    to intermediate data and finally to calculated descriptive features.
    Values are assigned automatically to ensure uniqueness and sequential order.
    """

    # //--- 1. Core File & Signal Properties ---//
    # Basic metadata and properties directly from the audio file.
    UUID = auto()
    AUDIO_PATH = auto()
    AUDIO_SAMPLES = auto()
    SAMPLE_RATE_HZ = auto()
    SAMPLE_RATE_KHZ = auto()
    DURATION = auto()
    BIT_DEPTH = auto()
    BIT_RATE = auto()
    NUM_CHANNELS = auto()
    DATA_RATE = auto()
    MAX_DATA_RATE = auto()
    DATA_EFFICIENCY = auto()
    EXISTING_TEMPO_CACHE = auto()

    # //--- 2. Intermediate Time-Series Data ---//
    # Raw or processed data arrays that are typically inputs for final feature calculation.
    HARMONIC_AUDIO = auto()
    PERCUSSIVE_AUDIO = auto()
    HARMONIC_MAGNITUDE_SPECTROGRAM = auto()
    ONSET_ENVELOPE = auto()
    PERCUSSIVE_ONSET_ENVELOPE = auto()
    ONSET_PEAKS = auto()
    PERCUSSIVE_ONSET_PEAKS = auto()
    MULTI_BAND_ONSET_ENVELOPES = auto()
    MULTI_BAND_ONSET_PEAKS = auto()
    BEAT_FRAMES = auto()
    BEAT_TIMES = auto()
    SPECTRAL_FLUX_ARRAY = auto()
    SPECTRAL_CENTROID_ARRAY = auto()
    LOUDNESS_ANALYSIS_RESULT = auto()
    MULTI_BAND_ENERGY = auto()
    FFT_FREQUENCIES = auto()
    CHROMAGRAM = auto()
    TRACK_FEATURE_VECTOR = auto()
    TRACK_SEGMENTS_RAW = auto()
    TRACK_SEGMENTS_PROFILED = auto()
    TRACK_NOTE_EVENTS = auto()
    TRACK_CLEANED_BINARY_MASK = auto()
    TRACK_MIDI_MAP = auto()
    TRACK_NORMALIZED_PITCH_CLASSES_MASK = auto()
    TRACK_PITCH_CLASSES = auto()
    SPECTRAL_PITCH_ARRAY = auto()
    SPECTRAL_MAGNITUDES_ARRAY = auto()
    TRACK_CLEANED_CHROMA_MASK = auto()

    # //--- 3. Rhythmic Features ---//
    # Features describing tempo, beats, and transient events.
    BPM = auto()
    DYNAMIC_TEMPO = auto()
    TEMPO_VARIATION = auto()
    BEAT_STRENGTH = auto()
    RHYTHMIC_REGULARITY = auto()
    ONSET_RATE_GLOBAL = auto()
    ONSET_ENV_MEAN_GLOBAL = auto()
    ONSET_RATE_KICK = auto()
    ONSET_ENV_MEAN_KICK = auto()
    ONSET_RATE_SNARE = auto()
    ONSET_ENV_MEAN_SNARE = auto()
    ONSET_RATE_LOW_MID = auto()
    ONSET_ENV_MEAN_LOW_MID = auto()
    ONSET_RATE_HI_HAT = auto()
    ONSET_ENV_MEAN_HI_HAT = auto()
    ONSET_RATE_VARIATION = auto()

    # //--- 4. Spectral Features ---//
    # Features describing the frequency content and timbre of the audio.
    SPECTRAL_CENTROID_MEAN = auto()
    SPECTRAL_CENTROID_STD = auto()
    SPECTRAL_CENTROID_MAX = auto()
    SPECTRAL_FLUX_MEAN = auto()
    SPECTRAL_FLUX_STD = auto()
    SPECTRAL_FLUX_MAX = auto()
    SPECTRAL_ROLLOFF_MEAN = auto()
    SPECTRAL_ROLLOFF_STD = auto()
    SPECTRAL_FLATNESS_MEAN = auto()
    SPECTRAL_FLATNESS_STD = auto()
    SPECTRAL_CONTRAST_MEAN = auto()
    SPECTRAL_CONTRAST_STD = auto()
    SPECTRAL_SKEWNESS = auto()
    SPECTRAL_KURTOSIS = auto()
    SPECTRAL_ENTROPY = auto()
    SPECTRAL_BANDWIDTH_MEAN = auto()
    SPECTRAL_BANDWIDTH_STD = auto()
    HARMONICITY = auto()
    HPR = auto()
    ZCR_MEAN = auto()
    ZCR_STD = auto()
    BASS_ENERGY_RATIO = auto()
    SUB_BASS_ENERGY_RATIO = auto()
    LOW_MID_ENERGY_RATIO = auto()
    MID_ENERGY_RATIO = auto()
    HIGH_ENERGY_RATIO = auto()

    # //--- 5. Loudness Features ---//
    # Features describing the perceived volume and dynamic range.
    INTEGRATED_LUFS = auto()
    INTEGRATED_LUFS_MEAN = auto()
    INTEGRATED_LUFS_STD = auto()
    INTEGRATED_LUFS_RANGE = auto()
    LOUDNESS_RANGE_LU = auto()
    TRUE_PEAK = auto()
    CREST_FACTOR = auto()
    RMS_MEAN = auto()
    RMS_MAX = auto()
    RMS_IQR = auto()
    RMS_PERC = auto()
    ENERGY_LEVEL = auto()

    # //--- 6. MFCC (Mel-Frequency Cepstral Coefficients) Features ---//
    # Standard features for timbre and speech/music recognition.
    MFCC_MEANS = auto()
    MFCC_STDS = auto()
    MFCC_VELOCITIES_MEANS = auto()
    MFCC_VELOCITIES_STDS = auto()
    MFCC_ACCELERATIONS_MEANS = auto()
    MFCC_ACCELERATIONS_STDS = auto()

    # //--- 7. Tonal & Harmonic Features ---//
    # Features related to musical key, chords, and pitch content.
    PRINCIPAL_KEY = auto()
    START_KEY = auto()
    END_KEY = auto()
    KEY_PROGRESSION = auto()
    CHROMA_ENTROPY = auto()

MFCC_UNIQUE_FILE_FEATURES: List[AudioDataFeature] = [
    AudioDataFeature.MFCC_MEANS, AudioDataFeature.MFCC_STDS,
    AudioDataFeature.MFCC_VELOCITIES_MEANS, AudioDataFeature.MFCC_VELOCITIES_STDS,
    AudioDataFeature.MFCC_ACCELERATIONS_MEANS, AudioDataFeature.MFCC_ACCELERATIONS_STDS
]

KEY_PROGRESSION_UNIQUE_FILE_FEATURES: List[AudioDataFeature] = [
    AudioDataFeature.KEY_PROGRESSION,
]

