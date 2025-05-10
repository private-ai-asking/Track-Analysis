import pickle
from pathlib import Path
import faiss
import numpy as np
from pandas import DataFrame
from sentence_transformers import SentenceTransformer

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.constants import DATA_DIRECTORY
from track_analysis.components.track_analysis.features.scrobbling.scrobble_data_loader import ScrobbleDataLoader


class EmbeddingBuilder:
    """Used to build embeddings for the library, refactored for clarity and SOLID principles."""
    def __init__(
            self,
            logger: HoornLogger,
            scrobble_data_loader: ScrobbleDataLoader,
            embedder: SentenceTransformer,
            combo_key: str = "||",
            sample_scrobbles: int = None
    ):
        self._logger: HoornLogger = logger
        self._separator: str = "EmbeddingBuilder"
        self._loader: ScrobbleDataLoader = scrobble_data_loader
        self._embedder: SentenceTransformer = embedder
        self._combo_key: str = combo_key
        self._sample_scrobbles: int = sample_scrobbles

        # Paths
        base = DATA_DIRECTORY / "__internal__"
        self._embeddings_path: Path = base / "lib_embeddings.npy"
        self._keys_path: Path = base / "lib_keys.pkl"
        self._index_path: Path = base / "lib.index"

        self._logger.trace("Successfully initialized.", separator=self._separator)

    def build_embeddings(self) -> None:
        """Main orchestration method to build and persist library embeddings and index."""
        self._logger.info("Starting to build embeddings...", separator=self._separator)

        data = self._load_library_data()
        combo_strings = self._build_combo_strings(data)
        embeddings = self._compute_embeddings(combo_strings)
        self._persist_embeddings_and_keys(embeddings, data)
        self._build_and_save_index(embeddings)

        self._logger.info("Done building embeddings.", separator=self._separator)

    def _load_library_data(self) -> DataFrame:
        """Load library data, using an optional sample for initialization."""
        self._loader.load(sample_rows=self._sample_scrobbles)
        return self._loader.get_library_data()

    def _build_combo_strings(self, library_data: DataFrame) -> list[str]:
        """Construct composite strings from artist, album, and title."""
        combo = self._combo_key
        return (
                library_data['_n_artist']
                + combo + library_data['_n_album']
                + combo + library_data['_n_title']
        ).tolist()

    def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for given texts using the embedder."""
        embs = self._embedder.encode(
            texts,
            batch_size=128,
            convert_to_numpy=True,
        ).astype('float32')
        return embs

    def _persist_embeddings_and_keys(self, embeddings: np.ndarray, library_data: DataFrame) -> None:
        """Save raw embeddings and corresponding UUID keys to disk."""
        np.save(self._embeddings_path, embeddings)
        with open(self._keys_path, 'wb') as f:
            pickle.dump(library_data['UUID'].tolist(), f)

    def _build_and_save_index(self, embeddings: np.ndarray) -> None:
        """Create a FAISS index for the embeddings and persist it."""
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        faiss.write_index(index, str(self._index_path))
