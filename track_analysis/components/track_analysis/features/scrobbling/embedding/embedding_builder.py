import pickle
from pathlib import Path

import faiss
import numpy as np
from pandas import DataFrame

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import DATA_DIRECTORY
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_data_loader import ScrobbleDataLoader
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_utility import ScrobbleUtility


class EmbeddingBuilder:
    """
    Build and persist a combined weighted FAISS index for title, artist, and album embeddings,
    delegating embedding logic to ScrobbleUtility for consistency.
    """

    def __init__(
            self,
            logger: HoornLogger,
            scrobble_data_loader: ScrobbleDataLoader,
            scrobble_utils: ScrobbleUtility,
            sample_scrobbles: int = None
    ):
        self._logger = logger
        self._sep = "EmbeddingBuilder"
        self._loader = scrobble_data_loader
        self._scrobble_utils = scrobble_utils
        self._sample_scrobbles = sample_scrobbles

        base = DATA_DIRECTORY / "__internal__"
        self._keys_path: Path = base / "lib_keys.pkl"
        self._combined_emb_path: Path = base / 'lib_emb_combined.npy'
        self._combined_index_path: Path = base / 'lib_combined.index'

        self._logger.trace("Initialized EmbeddingBuilderSingle.", separator=self._sep)

    def build_embeddings(self) -> None:
        """
        Load library data, delegate embedding computation, and save keys, embeddings, and FAISS index.
        """
        self._logger.info("Starting combined embedding build...", separator=self._sep)
        lib_df = self._load_library_data()
        self._persist_keys(lib_df)

        # extract normalized text fields
        titles = lib_df['_n_title'].astype(str).tolist()
        artists = lib_df['_n_artist'].astype(str).tolist()
        albums = lib_df['_n_album'].astype(str).tolist()

        # delegate to ScrobbleUtility for consistent embedding pipeline
        combined_emb = self._scrobble_utils.build_combined_embeddings(titles, artists, albums)

        # persist combined embeddings and index
        np.save(self._combined_emb_path, combined_emb)
        self._logger.debug(f"Saved combined embeddings to {self._combined_emb_path}", separator=self._sep)

        self._build_and_save_index(combined_emb, self._combined_index_path)
        self._logger.info("Completed combined embedding build.", separator=self._sep)

    def _load_library_data(self) -> DataFrame:
        """Load full or sampled library data."""
        self._loader.load(sample_rows=self._sample_scrobbles)
        return self._loader.get_library_data()

    def _persist_keys(self, library_data: DataFrame) -> None:
        """Save the ordered list of UUID keys once."""
        with open(self._keys_path, 'wb') as f:
            pickle.dump(library_data['UUID'].tolist(), f)
        self._logger.debug(f"Saved UUID keys to {self._keys_path}", separator=self._sep)

    def _build_and_save_index(self, embeddings: np.ndarray, path: Path) -> None:
        """Construct a FAISS L2 index on embeddings and write it to disk."""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, str(path))
        self._logger.debug(f"Wrote FAISS index to {path}", separator=self._sep)
