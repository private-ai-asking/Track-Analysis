import traceback

import numpy as np
import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.track_downloading.model.download_model import DownloadModel
from track_analysis.components.track_analysis.features.track_downloading.pipeline.download_pipeline_context import \
    DownloadPipelineContext


class GatherTracks(IPipe):
    """Pipe to gather all the tracks from the csv."""
    def __init__(self, logger: HoornLogger):
        self._logger = logger
        self._separator: str = "DownloadPipeline.GatherTracks"
        self._logger.trace("Successfully initialized.", separator=self._separator)

    def flow(self, data: DownloadPipelineContext) -> DownloadPipelineContext:
        if not data.download_csv_path.exists():
            self._logger.warning(f"The given path \"{data.download_csv_path}\" does not exist!", separator=self._separator)
            return data

        # noinspection PyBroadException
        try:
            df: pd.DataFrame = pd.read_csv(data.download_csv_path, header=0)
            df = df.replace({np.nan: None})
            records = df.to_dict(orient='records')

            tracks_to_download = [
                DownloadModel(
                    url=rec["URL"],
                    release_id=rec["RELEASE ID"],
                    recording_id=rec["TRACK ID"],
                    genre=rec["GENRE"],
                    subgenre=rec["SUBGENRES"]
                )
                for rec in records
            ]
            data.tracks_to_download = tracks_to_download
        except Exception:
            tb = traceback.format_exc()
            self._logger.error(
                f"Something went wrong while gathering tracks.\n{tb}",
                separator=self._separator
            )
        finally:
            return data
