from pathlib import Path
from typing import Optional, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import pydantic
from numpy import ndarray
from concurrent.futures import ThreadPoolExecutor
from pymediainfo import MediaInfo

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.apis.ffprobe_client import FFprobeClient
from track_analysis.components.track_analysis.features.key_extraction.utils.beat_detector import BeatDetector
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
    samples: Optional[ndarray] = None

    model_config = {
        "arbitrary_types_allowed": True
    }


class AudioFileHandler:
    """Handles audio file information retrieval using ffprobe and block-wise reading in parallel."""

    def __init__(
            self,
            logger: HoornLogger,
            block_size: int = 4096 * 128,
            num_workers: int = 4
    ):
        self._separator = "AudioFileHandler"
        self._logger = logger
        self._ffprobe_client = FFprobeClient(logger)
        self._audio_format_converter = AudioFormatConverter(logger)
        self._block_size = block_size
        self._num_workers = num_workers
        self._beat_detector: BeatDetector = BeatDetector(logger)

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def get_audio_streams_info_batch(
            self,
            audio_files: List[Path]
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
                    executor.map(self._extract_audio_info, audio_files),
                    start=1
            ):
                models.append(model)
                progress = idx / total * 100
                self._logger.info(
                    f"Processed {idx}/{total} ({progress:.2f}%)",
                    separator=self._separator
                )

        return models

    def _extract_audio_info(
            self,
            audio_file: Path
    ) -> AudioStreamsInfoModel:
        """Extracts metadata via MediaInfo and reads samples block-wise."""
        # 1) Fast metadata via MediaInfo
        media_info = MediaInfo.parse(str(audio_file))
        audio_track = media_info.audio_tracks[0]

        duration_s   = float(audio_track.duration) / 1000.0
        bitrate_bps  = float(audio_track.bit_rate) if audio_track.bit_rate is not None else 0
        sample_rate  = int(audio_track.sampling_rate)
        bit_depth    = float(audio_track.bit_depth) if audio_track.bit_depth else None
        channels     = int(audio_track.channel_s)
        audio_format = audio_track.format

        # 2) Block-wise sample reading
        samples, sr = librosa.load(audio_file, sr=None)

        # 3) Sanity-check sample rate
        if sr != sample_rate:
            self._logger.warning(
                f"Sample-rate mismatch for {audio_file}: "
                f"media_info={sample_rate} Hz vs librosa={sr} Hz",
                separator=self._separator
            )

        tempo, _, _ = self._beat_detector.detect(samples, sample_rate)

        # 4) Package into model
        return AudioStreamsInfoModel(
            duration        = duration_s,
            bitrate         = bitrate_bps / 1000,
            sample_rate_kHz = sample_rate / 1000,
            sample_rate_Hz  = sample_rate,
            bit_depth       = bit_depth,
            channels        = channels,
            format          = audio_format,
            samples         = samples,
            tempo           = tempo
        )

    def _read_samples_blockwise(
            self,
            audio_file: Path
    ) -> Tuple[ndarray, int]:
        """Read a file into a pre-allocated buffer using block-size frames."""
        with sf.SoundFile(str(audio_file), mode='r') as f:
            total_frames = f.frames
            channels = f.channels
            samplerate = f.samplerate

            buf = np.empty((total_frames, channels), dtype='float32')
            idx = 0
            for block in f.blocks(
                    blocksize=self._block_size,
                    always_2d=True,
                    dtype='float32'
            ):
                n = block.shape[0]
                buf[idx: idx + n] = block
                idx += n

        return buf, samplerate
