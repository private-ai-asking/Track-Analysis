from huggingface_hub import snapshot_download

from track_analysis.components.track_analysis.constants import DATA_DIRECTORY

model_path_cross: str = str(DATA_DIRECTORY / "__internal__" / "all-MiniLM-l6-v2-cross")

snapshot_download(repo_id="sentence-transformers/all-MiniLM-l6-v2", local_dir=model_path_cross)
