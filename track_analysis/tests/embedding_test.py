import pickle
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.testing import TestInterface
from track_analysis.components.md_common_python.py_common.user_input.user_input_helper import UserInputHelper
from track_analysis.components.md_common_python.py_common.utils import \
    gaussian_exponential_kernel_confidence_percentage
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_data_loader import ScrobbleDataLoader


class EmbeddingTest(TestInterface):
    """Used to test the embedding and matching strategy."""

    def __init__(self,
                 logger: HoornLogger,
                 embedder: SentenceTransformer,
                 keys_path: Path,
                 data_loader: ScrobbleDataLoader):
        super().__init__(logger, is_child=True)
        self._separator = "EmbeddingTest"
        self._embedding_model = embedder
        self._keys_path = keys_path
        self._data_loader = data_loader
        self._input_helper: UserInputHelper = UserInputHelper(logger, self._separator)
        self._logger.trace("Successfully initialized.", separator=self._separator)
        self._logger.warning("CURRENTLY NOT WORKING -> Needs to adapt to different embedding system.", separator=self._separator)



    def test(
            self,
            n_to_print: int = 5
    ) -> None:
        """Embed the key, search the index, and log the top-n results."""
        library_df: pd.DataFrame = self._data_loader.get_library_data()
        key_to_test: str = self._input_helper.get_user_input("Enter the key you want to test.", expected_response_type=str, validator_func=lambda s: (True, ""))

        index, _, __, ___ = self._data_loader.get_index()
        keys = self._load_keys()
        lookup = self._build_lookup(library_df)
        embedding = self._compute_embedding(key_to_test)
        distances, indices = self._search(index, embedding, n_to_print)
        results = self._assemble_results(distances[0], indices[0], keys, lookup)
        self._log_results(key_to_test, results)

    def _load_keys(self) -> list:
        self._logger.info(f"Loading keys from {self._keys_path}", separator=self._separator)
        with open(self._keys_path, 'rb') as f:
            return pickle.load(f)

    def _build_lookup(self, df: pd.DataFrame) -> dict:
        self._logger.debug("Building metadata lookup.", separator=self._separator)
        return {rec['UUID']: rec for rec in df.to_dict(orient='records')}

    def _compute_embedding(self, text: str) -> 'np.ndarray':
        self._logger.info(f"Embedding input: '{text}'", separator=self._separator)
        embed = self._embedding_model.encode(
            [text],
            convert_to_numpy=True,
            batch_size=1,
            show_progress_bar=False
        ).astype('float32')
        return embed

    def _search(
            self,
            index: faiss.Index,
            embedding: 'np.ndarray',
            top_k: int
    ) -> tuple:
        self._logger.info(f"Searching for top {top_k} nearest neighbors...", separator=self._separator)
        return index.search(embedding, top_k)

    def _assemble_results(
            self,
            distances: list,
            indices: list,
            keys: list,
            lookup: dict
    ) -> list:
        results = []
        for rank, (dist, idx) in enumerate(zip(distances, indices), start=1):
            uuid = keys[idx] if idx < len(keys) else '<UNKNOWN>'
            meta = lookup.get(uuid, {})
            results.append({
                'rank': rank,
                'uuid': uuid,
                'title': meta.get('_n_title', 'Unknown Title'),
                'album': meta.get('_n_album', 'Unknown Album'),
                'artist': meta.get('_n_artist', 'Unknown Artist'),
                'distance': dist
            })
        return results

    def _log_results(self, query: str, results: list) -> None:
        self._logger.info(f"Nearest neighbors for '{query}':", separator=self._separator)
        for r in results:
            distance: float = r['distance']

            self._logger.info(
                f"{r['rank']}. [{r['uuid']}] {r['title']} — {r['album']} — {r['artist']} "
                f"(distance: {distance:.4f}) [confidence: {gaussian_exponential_kernel_confidence_percentage(distance, sigma=0.35):.4f}]",
                separator=self._separator
            )
