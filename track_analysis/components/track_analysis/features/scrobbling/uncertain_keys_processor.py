import json
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.user_input.user_input_helper import UserInputHelper
from track_analysis.components.track_analysis.constants import CACHE_DIRECTORY
from track_analysis.components.track_analysis.features.scrobbling.embedding.embedding_searcher import EmbeddingSearcher
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_data_loader import ScrobbleDataLoader
from track_analysis.components.track_analysis.features.scrobbling.utils.scrobble_utility import ScrobbleUtility


class UncertainKeysProcessor:
    """Interactively process uncertain scrobbles and update overrides."""
    def __init__(
            self,
            logger: HoornLogger,
            embedding_searcher: EmbeddingSearcher,
            scrobble_utility: ScrobbleUtility,
            data_loader: ScrobbleDataLoader,
            manual_override_json_path: Path
    ):
        self._logger = logger
        self._separator = "UncertainKeysProcessor"
        self._input = UserInputHelper(logger, self._separator)
        self._searcher = embedding_searcher
        self._utils = scrobble_utility
        self._loader = data_loader
        self._override_path = manual_override_json_path

        self._top_k = self._searcher.get_top_k_num()
        self.accepted: List[Tuple[str, str]] = []
        self.rejected: List[str] = []

        self._logger.trace("Initialized.", separator=self._separator)

    @staticmethod
    def _get_uncertain_df() -> pd.DataFrame:
        return pd.read_csv(CACHE_DIRECTORY / "uncertain_keys_temp.csv")

    def process(self) -> List[Tuple[str, Union[str, None]]]:
        df = self._get_uncertain_df()
        if df.empty:
            print("No uncertain scrobbles to review.")
            return []

        # Load library
        self._loader.load()
        library_index = self._loader.get_index()
        library_keys = self._loader.get_keys()
        library_rows = (
            self._loader.get_library_data()[["UUID", "_n_title", "_n_artist", "_n_album"]]
            .set_index("UUID")
            .to_dict(orient="index")
        )

        # Sort
        df = df.sort_values("__key").reset_index(drop=True)
        total = len(df)

        for idx, row in df.iterrows():
            key = row["__key"]
            title = row.get("_n_title", "")
            artist = row.get("_n_artist", "")
            album = row.get("_n_album", "")

            print(f"\n[{idx+1}/{total}] Scrobble key: {key}")
            print(f"    Title : {title}")
            print(f"    Artist: {artist}")
            print(f"    Album : {album}")

            emb = self._utils.build_combined_embeddings([title], [artist], [album])
            indices, distances = self._searcher._search(emb, library_index)
            indices = indices[0]
            distances = distances[0]

            print("Top candidates:")
            for rank, (lib_idx, dist) in enumerate(zip(indices, distances), start=1):
                uuid = library_keys[lib_idx]
                meta = library_rows.get(uuid, {})
                print(f"  [{rank}] {uuid} ({dist:.4f})")
                print(f"       Title : {meta.get('_n_title','')}")
                print(f"       Artist: {meta.get('_n_artist','')}")
                print(f"       Album : {meta.get('_n_album','')}")

            # Use UserInputHelper: expect a str, validator allows 'q' or digit in range
            prompt = (
                f"Enter 1–{self._top_k} to accept, 0 to reject, or 'q' to quit and save: "
            )
            def validator(inp: str):
                if inp.lower() == 'q':
                    return True, ""
                if inp.isdigit() and 0 <= int(inp) <= self._top_k:
                    return True, ""
                return False, f"must be 'q' or an integer 0–{self._top_k}"

            choice_raw: str = self._input.get_user_input(
                prompt=prompt,
                expected_response_type=str,
                validator_func=validator
            )

            if choice_raw.lower() == 'q':
                print("Quitting early; saving progress…")
                break

            choice = int(choice_raw)
            if choice == 0:
                print(f"Rejected {key}.")
                self.rejected.append(key)
            else:
                sel_uuid = library_keys[indices[choice - 1]]
                print(f"Accepted {key} → {sel_uuid}")
                self.accepted.append((key, sel_uuid))

        # Remove processed from uncertain_keys_temp.csv
        processed = {k for k, _ in self.accepted} | set(self.rejected)
        df_left = df[~df["__key"].isin(processed)]
        df_left.to_csv(CACHE_DIRECTORY / "uncertain_keys_temp.csv", encoding='utf-8')

        # Merge into manual-override JSON
        overrides = {}
        if self._override_path.is_file():
            overrides = json.loads(self._override_path.read_text())
        for key, uuid in self.accepted:
            overrides[key] = uuid
        for key in self.rejected:
            overrides[key] = None

        self._override_path.write_text(json.dumps(overrides, indent=2, sort_keys=True))

        # Return list of (key, uuid_or_None)
        return [(k, u) for k, u in self.accepted] + [(k, None) for k in self.rejected]
