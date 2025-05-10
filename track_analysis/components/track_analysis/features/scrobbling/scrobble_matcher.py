import pandas as pd

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import NO_MATCH_LABEL
from track_analysis.components.track_analysis.features.scrobbling.scrobble_utility import ScrobbleUtility


class ScrobbleMatcher:
    """
    Matches scrobbles to library tracks via a pre-built cache only.
    """

    def __init__(self, logger: HoornLogger, cache_builder: CacheBuilder, scrobble_utils: ScrobbleUtility):
        self._logger = logger
        self._cache = cache_builder
        self._scrobble_utils = scrobble_utils

    def link_scrobbles(self, scrobble_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each scrobble record, look up a match only in the built cache.
        Unmatched items get NO_MATCH_LABEL.
        """
        records = scrobble_df.to_dict(orient='records')
        texts = [self._scrobble_utils.compute_key(r['_n_title'], r['_n_artist'], r['_n_album']) for r in records]

        uuids = []
        confidences = []
        for txt in texts:
            match = self._cache.get(txt)
            if match is not None:
                uuids.append(match)
            else:
                uuids.append(NO_MATCH_LABEL)
            confidences.append(None)
            # update cache to keep consistency (no new entries)

        output = scrobble_df.copy()
        output['track_uuid'] = uuids
        output['confidence'] = confidences

        return output
