import pandas as pd

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_data_loader import ScrobbleDataLoader
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_utility import ScrobbleUtility
from track_analysis.components.track_analysis.library.configuration.model.configuration import \
    TrackAnalysisConfigurationModel


class ScrobbleMatcher:
    """
    Matches scrobbles to library tracks via a pre-built cache only.
    """

    def __init__(self,
                 logger: HoornLogger,
                 cache_builder: CacheBuilder,
                 scrobble_utils: ScrobbleUtility,
                 data_loader: ScrobbleDataLoader,
                 app_config: TrackAnalysisConfigurationModel):
        self._logger = logger
        self._cache = cache_builder
        self._scrobble_utils = scrobble_utils
        self._loader = data_loader
        self._app_config = app_config

    def link_scrobbles(self, scrobble_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each scrobble record, look up a match only in the built cache.
        Unmatched items get NO_MATCH_LABEL.
        """
        records = scrobble_df.to_dict(orient='records')
        texts = [self._scrobble_utils.compute_key(r['_n_title'], r['_n_artist'], r['_n_album']) for r in records]
        txt_record_lookup = {
            txt: rec for txt, rec in zip(texts, records)
        }

        uuids = []
        primary_artists = []
        fallbacks = []
        for txt in texts:
            matching_cache_entry = self._cache.get(txt)

            if matching_cache_entry is not None:
                uuid = matching_cache_entry["associated_uuid"]
                matching_row = self._loader.get_library_row_by_uuid_lookup()[uuid]
                primary_artist = matching_row["Primary Artist"]

                uuids.append(uuid)
                primary_artists.append(primary_artist)
                fallbacks.append(primary_artist)
            else:
                uuids.append(self._app_config.additional_config.no_match_label)
                primary_artists.append(None)
                fallbacks.append(txt_record_lookup[txt]["Artist(s)"])

        output = scrobble_df.copy()
        output['track_uuid'] = uuids
        output['Primary Artist'] = primary_artists

        output['Artist Distinguish'] = fallbacks

        return output
