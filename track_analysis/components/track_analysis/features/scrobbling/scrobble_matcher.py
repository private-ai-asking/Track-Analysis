import pandas as pd

from track_analysis.components.md_common_python.py_common.cache_helpers import CacheBuilder
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import NO_MATCH_LABEL


class ScrobbleMatcher:
    """
    Matches scrobbles to library tracks via a pre-built cache only.
    """

    def __init__(self, logger: HoornLogger, cache_builder: CacheBuilder):
        self._logger = logger
        self._cache = cache_builder

    def _create_key(self, artist: str, album: str, title: str) -> str:
        return f"{artist}||{album}||{title}"

    def link_scrobbles(self, scrobble_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each scrobble record, look up a match only in the built cache.
        Unmatched items get NO_MATCH_LABEL.
        """
        records = scrobble_df.to_dict(orient='records')
        texts = [self._create_key(r['_n_artist'], r['_n_album'], r['_n_title']) for r in records]

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
