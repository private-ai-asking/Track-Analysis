from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.tag_extractor import TagExtractor


class GetAndBuildAudioMetadata(IPipe):
    def __init__(self,
                 logger: HoornLogger,
                 tag_extractor: TagExtractor):
        self._separator = "BuildCSV.GetAndBuildAudioMetadata"
        self._logger    = logger
        self._tag_extractor = tag_extractor
        self._logger.trace("Initialized combined pipe.", separator=self._separator)

    def flow(self, data: LibraryDataGenerationPipelineContext) -> LibraryDataGenerationPipelineContext:
        paths = data.filtered_audio_file_paths
        total = len(paths)
        self._logger.trace("Beginning audio metadata extraction…", separator=self._separator)
        self._logger.info(f"Paths to extract metadata: {total}", separator=self._separator)

        # 1. Helper that returns a dict of metadata for one file
        def extract_one(path: str) -> dict:
            s = pd.Series({ Header.Audio_Path.value: path })
            self._tag_extractor.add_extracted_metadata_to_track(s)
            return s.to_dict()

        # 2. Parallelize the I/O-bound tag extraction
        records = []
        with ThreadPoolExecutor() as executor:
            for idx, rec in enumerate(executor.map(extract_one, paths), start=1):
                records.append(rec)
                self._logger.info(
                    f"Processed {idx}/{total} ({idx/total*100:.4f}%) tracks.",
                    separator=self._separator
                )

        # 3. Build your DataFrame in one shot
        if len(records) > 0:
            df = pd.DataFrame.from_records(records)
            data.generated_audio_info = df

            # 4. Vectorized album-cost lookup
            album_cost_map = {info.Album_Title: info.Album_Cost for info in data.album_costs}
            df[Header.Album_Cost.value] = (
                df[Header.Album.value]
                .map(album_cost_map)
                .fillna(0)
            )
        else:
            data.generated_audio_info = pd.DataFrame()

        self._logger.info("Successfully extracted all audio metadata.", separator=self._separator)
        return data
