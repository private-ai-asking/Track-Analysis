from typing import Dict

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.audio_file_handler import AudioFileHandler
from track_analysis.components.track_analysis.features.core.cacheing.mfcc import MfccExtractor
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.data_generation.redo_headers.base_header_processor import \
    BaseHeaderProcessor
from track_analysis.components.track_analysis.features.data_generation.redo_headers.bpm_processor import BPMProcessor
from track_analysis.components.track_analysis.features.data_generation.redo_headers.energy_processor import \
    EnergyProcessor
from track_analysis.components.track_analysis.features.data_generation.redo_headers.key_processor import KeyProcessor
from track_analysis.components.track_analysis.features.data_generation.redo_headers.mfcc_processor import MFCCProcessor
from track_analysis.components.track_analysis.features.data_generation.util.key_extractor import KeyExtractor


class RedoHeaders(IPipe):
    """
    Orchestrator class that delegates header processing to specialized processor classes.
    """

    _SEPARATOR = "BuildCSV.RedoHeaders"

    def __init__(self, logger: HoornLogger, file_handler: AudioFileHandler, num_workers: int):
        self._logger = logger
        self._file_handler = file_handler
        self._num_workers = num_workers
        self._key_extractor = KeyExtractor(logger, file_handler, num_workers)

        # Initialize the processor strategies
        self._bpm_processor = BPMProcessor(logger, file_handler)
        self._key_processor = KeyProcessor(logger, file_handler, self._key_extractor)
        self._energy_processor = EnergyProcessor(logger, file_handler)
        self._mfcc_processor = MFCCProcessor(logger, file_handler, MfccExtractor(logger))

        self._header_processors: Dict[Header, BaseHeaderProcessor] = {
            Header.BPM: self._bpm_processor,
            Header.Key: self._key_processor,
            Header.Energy_Level: self._energy_processor,
            Header.MFCC: self._mfcc_processor,
        }

        self._logger.trace("Successfully initialized pipe.", separator=self._SEPARATOR)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        """
        Executes the header refill logic by delegating to the appropriate processor.
        """
        for header, uuids in data.refill_headers.items():
            processor = self._header_processors.get(header)
            if not processor:
                self._logger.warning(
                    f"Header {header} not found, unsupported refill... skipping.",
                    separator=self._SEPARATOR
                )
                continue
            processor.process(uuids, data)
        return data
