from typing import Tuple, List, Dict

import faiss
import numpy as np

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.scrobbling.embedding.candidate_retriever_interface import \
    CandidateRetrieverInterface
from track_analysis.components.track_analysis.features.scrobbling.model.candidate_model import CandidateModel
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_data_loader import ScrobbleDataLoader
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_utility import ScrobbleUtility


class EmbeddingSearcher:
    """Utility class for searching embeddings."""
    def __init__(self,
                 logger: HoornLogger,
                 top_k: int,
                 loader: ScrobbleDataLoader,
                 utility: ScrobbleUtility,
                 candidate_retriever: CandidateRetrieverInterface):
        self._logger = logger
        self._separator: str = "EmbeddingSearcher"

        self._top_k = top_k
        self._loader = loader
        self._utils = utility
        self._candidate_retriever = candidate_retriever

        # Pre-allocate FAISS output buffers
        n_queries = 10_000
        self._search_indices_buffer = np.empty((n_queries, self._top_k), dtype=np.int64)
        self._search_distances_buffer = np.empty((n_queries, self._top_k), dtype=np.float32)

        self._logger.trace("Initialized Successfully.", separator=self._separator)

    def get_top_k_num(self) -> int:
        return self._top_k

    def search_batch(self, n_titles: List[str], n_albums: List[str], n_artists: List[str]) -> List[List[CandidateModel]]:
        self._loader.load()
        library_index: faiss.Index = self._loader.get_index()
        embeddings = self._utils.build_combined_embeddings(n_titles, n_artists, n_albums)
        indices, distances = self._search(embeddings, library_index)

        records: List[Dict[str, str]] = [
            {'_n_title': _n_title, '_n_artist': _n_artist, '_n_album': _n_album}
            for _n_title, _n_artist, _n_album in zip(n_titles, n_artists, n_albums)
        ]

        candidates_2d_array: List[List[CandidateModel]] = []

        for rec, rec_indices, rec_distances in zip(records, indices, distances):
            record_candidates: List[CandidateModel] = self._candidate_retriever.retrieve_candidates(
                rec, rec_indices, rec_distances
            )
            candidates_2d_array.append(record_candidates)

        return candidates_2d_array


    def search(self, n_title: str, n_album: str, n_artist: str) -> List[CandidateModel]:
        """Searches for Top K similar candidates based on the candidate retriever process."""
        self._loader.load()
        library_index: faiss.Index = self._loader.get_index()
        embeddings = self._utils.build_combined_embeddings([n_title], [n_artist], [n_album])

        indices, distances = self._search(embeddings, library_index)
        indices = indices[0]
        distances = distances[0]

        candidates: List[CandidateModel] = self._candidate_retriever.retrieve_candidates(
            record={'_n_title': n_title, '_n_album': n_album, '_n_artist': n_artist},
            neighbour_indices=indices,
            distances=distances
        )

        return candidates

    def _search(self, embeddings: np.ndarray, library_index: faiss.Index) -> Tuple[np.array, np.array]:
        if embeddings.dtype != np.float32 or not embeddings.flags["C_CONTIGUOUS"]:
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        nq = embeddings.shape[0]
        Dret, Iret = library_index.search(embeddings, self._top_k)
        self._search_distances_buffer[:nq] = Dret
        self._search_indices_buffer[:nq] = Iret
        return self._search_indices_buffer[:nq], self._search_distances_buffer[:nq]
