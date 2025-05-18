from typing import Dict, List

import numpy
from rapidfuzz import fuzz

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.utils import SimilarityScorer
from track_analysis.components.track_analysis.features.scrobbling.embedding.candidate_retriever_interface import \
    CandidateRetrieverInterface
from track_analysis.components.track_analysis.features.scrobbling.model.candidate_model import CandidateModel
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_data_loader import ScrobbleDataLoader


class DefaultCandidateRetriever(CandidateRetrieverInterface):
    def __init__(self,
                 logger: HoornLogger,
                 loader: ScrobbleDataLoader,
                 token_similarity_scorer: SimilarityScorer):
        super().__init__(logger, loader, is_child=True)
        self._separator = "DefaultCandidateRetriever"
        self._scorer = token_similarity_scorer

        self._logger.trace("Successfully initialized.", separator=self._separator)


    def retrieve_candidates(self, record: Dict, neighbour_indices: List[int], distances: numpy.array) \
            -> List[CandidateModel]:
        rec_title, rec_artist, rec_album = record["_n_title"], record["_n_artist"], record["_n_album"]
        score_fn = self._scorer.score
        lib_keys = self._loader.get_keys()
        library_row_lookup = self._loader.get_library_row_by_uuid_lookup()
        n_lib = len(lib_keys)

        candidates: List[CandidateModel] = []
        for lib_idx, distance in zip(neighbour_indices, distances):
            if not (0 <= lib_idx < n_lib):
                continue
            uuid = lib_keys[lib_idx]
            row = library_row_lookup.get(uuid)
            if not row:
                continue

            combined_sim = score_fn(
                {"title": rec_title, "artist": rec_artist, "album": rec_album},
                {"title": row["_n_title"], "artist": row["_n_artist"], "album": row["_n_album"]},
                optimize=True
            ) / 100.0

            sim_title  = fuzz.ratio(rec_title,  row["_n_title"])  / 100.0
            sim_artist = fuzz.ratio(rec_artist, row["_n_artist"]) / 100.0
            sim_album  = fuzz.ratio(rec_album,  row["_n_album"])  / 100.0

            candidates.append(CandidateModel(
                lib_idx      = lib_idx,
                uuid         = uuid,
                distance     = float(distance),
                combined_token_similarity = combined_sim,
                title_token_similarity    = sim_title,
                artist_token_similarity   = sim_artist,
                album_token_similarity    = sim_album,
            ))

        return candidates
