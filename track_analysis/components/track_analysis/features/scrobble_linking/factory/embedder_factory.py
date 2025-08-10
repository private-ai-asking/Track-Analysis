from pathlib import Path

from sentence_transformers import SentenceTransformer


class EmbedderFactory:
    """Creates the embedder."""
    def __init__(self, embedding_path: Path):
        self._embedding_path = embedding_path

    def create(self) -> SentenceTransformer:
        return SentenceTransformer(model_name_or_path=str(self._embedding_path), device="cuda")
