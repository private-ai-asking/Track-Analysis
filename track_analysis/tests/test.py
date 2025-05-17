from huggingface_hub import snapshot_download

from track_analysis.components.track_analysis.constants import DATA_DIRECTORY

download_path: str = str(DATA_DIRECTORY / "__internal__" / "all-MiniLM-l12-v2-embed")

snapshot_download(repo_id="sentence-transformers/all-MiniLM-l12-v2", local_dir=download_path)
