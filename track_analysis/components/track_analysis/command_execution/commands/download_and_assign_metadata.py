from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns.command.command_interface import T, \
    P
from track_analysis.components.track_analysis.command_execution.command_execution_model import CommandExecutionModel
from track_analysis.components.track_analysis.features.track_downloading.api.metadata_api import MetadataAPI
from track_analysis.components.track_analysis.features.track_downloading.model.download_model import DownloadModel
from track_analysis.components.track_analysis.features.track_downloading.pipeline.download_pipeline import \
    DownloadPipeline
from track_analysis.components.track_analysis.features.track_downloading.pipeline.download_pipeline_context import \
    DownloadPipelineContext
from track_analysis.components.track_analysis.library.configuration.model.configuration import \
    TrackAnalysisConfigurationModel


class DownloadAndAssignMetadataCommand(CommandExecutionModel):
    def __init__(self, logger: HoornLogger, download_pipeline: DownloadPipeline, metadata_api: MetadataAPI, configuration: TrackAnalysisConfigurationModel):
        super().__init__(logger)
        self._download_pipeline = download_pipeline
        self._metadata_api = metadata_api
        self._configuration = configuration

    @property
    def default_command_keys(self) -> List[str]:
        return ["download_and_md", "damd"]

    @property
    def command_description(self) -> str:
        return "Combines downloading and setting metadata."

    def execute(self, arguments: T) -> P:
        data: DownloadPipelineContext = self._download_pipeline.flow(
            DownloadPipelineContext(download_csv_path=self._configuration.paths.download_csv_file)
        )
        download_files: List[DownloadModel] = data.downloaded_tracks
        max_workers = min(32, len(download_files) or 1)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_track = {
                executor.submit(
                    self._metadata_api.populate_metadata_from_musicbrainz_for_file,
                    track
                ): track
                for track in download_files
            }

            for future in as_completed(future_to_track):
                track = future_to_track[future]
                try:
                    future.result()
                except Exception as e:
                    self._logger.error(
                        f"Error populating metadata for {track}: {e}"
                    )
