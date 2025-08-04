from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Dict

import librosa
import pydantic
from numpy import ndarray
from pymediainfo import MediaInfo

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.apis.ffprobe_client import FFprobeClient
from track_analysis.components.track_analysis.features.audio_calculation.utils.cacheing.max_rate_cache import \
    MaxRateCache
from track_analysis.components.track_analysis.features.audio_calculation.utils.file_utils import FileUtils
from track_analysis.components.track_analysis.features.core.cacheing.beat import BeatDetector
from track_analysis.components.track_analysis.util.audio_format_converter import AudioFormatConverter


class AudioStreamsInfoModel(pydantic.BaseModel):
    duration: float = 0
    bitrate: float = 0
    sample_rate_kHz: float = 0
    sample_rate_Hz: int = 0
    bit_depth: Optional[float] = 0
    channels: int = 0
    tempo: float = 0
    format: str
    path: Path
    samples: Optional[ndarray] = None

    actual_data_rate_kbps: float = 0
    max_data_per_second_kbps: float = 0
    efficiency: float = 0

    model_config = {
        "arbitrary_types_allowed": True
    }


class AudioFileHandler:
    """Handles audio file information retrieval using ffprobe and block-wise reading in parallel."""

    def __init__(
            self,
            logger: HoornLogger,
            max_rate_cache_path: Path,
            num_workers: int = 4
    ):
        self._separator = "AudioFileHandler"
        self._logger = logger
        self._ffprobe_client = FFprobeClient(logger)
        self._audio_format_converter = AudioFormatConverter(logger)
        self._num_workers = num_workers
        self._beat_detector: BeatDetector = BeatDetector(logger)
        self._file_utils: FileUtils = FileUtils()
        self._rate_cache: MaxRateCache = MaxRateCache(max_rate_cache_path)

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def get_audio_streams_info_batch(
            self,
            audio_files: List[Path],
            existing_tempos: Optional[Dict[Path, float]] = None
    ) -> List[AudioStreamsInfoModel]:
        """
        Process a batch of audio files in parallel, each read in block-wise chunks.
        Returns the list of AudioStreamsInfoModel in the same order as input files.
        """
        total = len(audio_files)
        models: List[AudioStreamsInfoModel] = []

        # Use ThreadPoolExecutor to parallelize I/O-bound block reads
        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            for idx, model in enumerate(
                    executor.map(
                        lambda p: self._extract_audio_info(p, existing_tempos),
                        audio_files
                    ),
                    start=1
            ):
                models.append(model)
                progress = idx / total * 100
                self._logger.info(
                    f"Processed {idx}/{total} ({progress:.2f}%)",
                    separator=self._separator
                )

        return models

    # noinspection t
    def _extract_audio_info(
            self,
            audio_file: Path,
            existing_tempos: Optional[Dict[Path, float]]
    ) -> AudioStreamsInfoModel:
        """Extracts metadata, samples, and file-based metrics efficiently."""
        # 1) Fast metadata via MediaInfo
        media_info = MediaInfo.parse(str(audio_file))
        audio_track = media_info.audio_tracks[0]

        duration_s = float(audio_track.duration) / 1000.0 if audio_track.duration else 0.0
        bitrate_bps = float(audio_track.bit_rate) if audio_track.bit_rate is not None else 0.0
        sample_rate = int(audio_track.sampling_rate) if audio_track.sampling_rate else 0
        bit_depth = float(audio_track.bit_depth) if audio_track.bit_depth else None
        channels = int(audio_track.channel_s) if audio_track.channel_s else 0
        audio_format = audio_track.format

        # 2) Load Audio
        samples, sr = librosa.load(audio_file, sr=None)
        if sr != sample_rate:
            self._logger.warning(
                f"Sample-rate mismatch for {audio_file}: "
                f"media_info={sample_rate} Hz vs librosa={sr} Hz",
                separator=self._separator
            )

        # 3) Get Tempo
        if existing_tempos and audio_file in existing_tempos:
            tempo = existing_tempos[audio_file]
        else:
            tempo, _, _ = self._beat_detector.detect(audio_path=audio_file, audio=samples, sample_rate=sr)

        # 4) Calculate File-Based Metrics using existing variables
        actual_rate_bps = (self._file_utils.get_size_bytes(audio_file) * 8) / duration_s if duration_s > 0 else 0.0

        max_rate_bps = self._rate_cache.get(sample_rate, bit_depth, channels)
        efficiency = (actual_rate_bps / max_rate_bps * 100) if max_rate_bps > 0 else 0.0

        # 5) Package into model using the clean local variables
        return AudioStreamsInfoModel(
            duration=duration_s,
            bitrate=bitrate_bps / 1000,
            sample_rate_kHz=sample_rate / 1000,
            sample_rate_Hz=sample_rate,
            bit_depth=bit_depth,
            channels=channels,
            format=audio_format,
            samples=samples,
            tempo=tempo,
            path=audio_file,
            actual_data_rate_kbps=actual_rate_bps / 1000,
            max_data_per_second_kbps=max_rate_bps / 1000,
            efficiency=efficiency
        )
