from pathlib import Path
from typing import List

from track_analysis.components.track_analysis.features.core.memory_utils import trim_and_defrag
from track_analysis.components.track_analysis.features.data_generation.model.header import Header
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import \
    LibraryDataGenerationPipelineContext
from track_analysis.components.track_analysis.features.data_generation.redo_headers.base_header_processor import \
    BaseHeaderProcessor


class BPMProcessor(BaseHeaderProcessor):
    def process(self, uuids: List[str], data: LibraryDataGenerationPipelineContext) -> None:
        df = data.loaded_audio_info_cache
        items = self._collect_index_path_pairs(df, uuids)
        batch_size = data.max_new_tracks_per_run
        processed_count = 0

        for start in range(0, len(items), batch_size):
            batch = items[start: start + batch_size]
            indices, paths = zip(*batch) if batch else ([], [])
            if not paths:
                continue

            bpms = self._fetch_bpms(list(paths))
            self._update_dataframe_column(df, list(indices), Header.BPM, bpms)
            self._write_bpm_tags(list(paths), bpms)
            processed_count += len(batch)

            self._logger.info(
                f"Completed batch of {len(batch)} BPM updates. Running cleanup.",
                separator=self._SEPARATOR
            )

            trim_and_defrag(self._logger, f"after BPM batch starting at index {start}")

        self._logger.info(
            f"Finished processing {processed_count} tracks with redoing BPM values!",
            separator=self._SEPARATOR
        )

    def _fetch_bpms(self, paths: List[Path]) -> List[float]:
        infos = self._log_and_handle_exception(
            "Failed to fetch BPMs for batch",
            self._file_handler.get_audio_streams_info_batch,
            paths
        )
        return [info.tempo for info in infos] if infos else [0.0] * len(paths)

    def _write_bpm_tags(self, paths: List[Path], bpms: List[float]) -> None:
        string_bpms = [str(round(bpm, 4)) for bpm in bpms]
        self._update_audio_tags(paths, "bpm", string_bpms)
