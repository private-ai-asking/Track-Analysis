# import numpy as np
# import pickle
# import pandas as pd
# import faiss
#
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
#
# from ace_tools_open import display_dataframe_to_user
#
# from track_analysis.components.md_common_python.py_common.logging import HoornLogger
# from track_analysis.components.md_common_python.py_common.utils import StringUtils
# from track_analysis.components.track_analysis.constants import DATA_DIRECTORY, OUTPUT_DIRECTORY
#
# # Paths to your saved artifacts
# data_dir = DATA_DIRECTORY / "__internal__"
# embeddings_path = data_dir / "lib_embeddings.npy"
# keys_path = data_dir / "lib_keys.pkl"
# index_path = data_dir / "lib.index"
# library_csv_path = OUTPUT_DIRECTORY / "data.csv"
#
# # Load embeddings and keys
# embeddings = np.load(embeddings_path).astype('float32')
# with open(keys_path, "rb") as f:
#     keys = pickle.load(f)
#
# # Load FAISS index
# index = faiss.read_index(str(index_path))
#
# # 1) Sanity: total vectors
# print(f"Index.ntotal = {index.ntotal}, embeddings.shape[0] = {embeddings.shape[0]}")
# assert index.ntotal == embeddings.shape[0]
#
# # 2) Sanity: IVF trained?
# if hasattr(index, "is_trained"):
#     print("Index is_trained:", bool(index.is_trained))
#     assert index.is_trained
#
# # Load original library data
# lib_df = pd.read_csv(library_csv_path)
#
# string_utils: StringUtils = StringUtils(logger=HoornLogger(outputs=None))
# lib_df["_n_title"] = lib_df["Title"].map(string_utils.normalize_field)
# lib_df["_n_artist"] = lib_df["Artist(s)"].map(string_utils.normalize_field)
# lib_df["_n_album"] = lib_df["Album"].map(string_utils.normalize_field)
#
# # Build normalized string column (same as you used to embed)
# combo = "||"
# lib_df["_combo_str"] = (
#         lib_df["_n_artist"] + combo +
#         lib_df["_n_album"]  + combo +
#         lib_df["_n_title"]
# )
#
# # Create inspection DataFrame
# inspect_df = pd.DataFrame({
#     "uuid": keys,
#     "artist": lib_df["_n_artist"],
#     "album": lib_df["_n_album"],
#     "title": lib_df["_n_title"],
#     "combo_string": lib_df["_combo_str"],
#     "vector_index": list(range(len(keys))),
# })
#
# # Show first 20 rows
# display_dataframe_to_user("Library Embedding Mapping", inspect_df.head(20))
#
# # 3) Self-query sanity check: vector should find itself as nearest neighbor
# N, d = embeddings.shape
# print("\nSelf-query checks:")
# for i in [0, N//2, N-1]:
#     vec = embeddings[i].reshape(1, d)
#     distances, indices = index.search(vec, k=3)
#     print(f"Query idx {i} -> returned indices {indices[0]}, distances {distances[0]}")
#     assert indices[0][0] == i and distances[0][0] < 1e-6
#
# # 4) Sample scrobble query (optional)
# sample_text = lib_df["_combo_str"].iloc[0]
# sample_vec = np.expand_dims(StringUtils(logger=HoornLogger(outputs=None)).normalize_field(sample_text), 0)  # reuse embed step if needed
# # Assuming _embed is available in your context:
# # sample_vec = _embed(sample_text).astype('float32').reshape(1, -1)
# # distances, indices = index.search(sample_vec, k=5)
# # print("\nSample scrobble query results:", indices, distances)
#
# print("\nAll sanity checks passed!")
# from huggingface_hub import snapshot_download
#
# from track_analysis.components.track_analysis.constants import DATA_DIRECTORY
#
# path = DATA_DIRECTORY / "__internal__" / "all-MiniLM-L6-v2"
#
# snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir=path)

