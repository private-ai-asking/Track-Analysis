import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.scrobble_linking.algorithm.algorithm_context import CacheBuildingAlgorithmContext


class ValidateManualOverride(IPipe):
    """A pipe to filter out the tracks present in the manual override."""
    def __init__(self, logger: HoornLogger, manual_override_json_path: Path):
        self._logger = logger
        self._separator = "CacheBuilder.ValidateManualOverride"
        self._override_path: Path = manual_override_json_path
        self._lookup = self._build_lookup()

        self._logger.trace(f"Loaded {len(self._lookup)} manual overrides.",
                           separator=self._separator)

    def flow(self, ctx: CacheBuildingAlgorithmContext) -> CacheBuildingAlgorithmContext:
        """Apply any manual overrides: accept, reject, or leave for later."""
        self._logger.debug("Validate manual override.", separator=self._separator)

        # 1. Find UUIDs that aren’t in the library
        missing_uuids = self._get_missing_uuids(
            ctx.library_data_frame,
            [override_uuid for _, override_uuid in self._lookup.items()]
        )

        if missing_uuids:
            # 2. Remove any lookup entries whose value is in missing_uuids
            for key, val in dict(self._lookup).items():  # iterate over a copy
                if val is not None and val in missing_uuids:
                    # 3. Warn about each invalid override
                    self._logger.warning(
                        f"Discarding manual override '{key}': "
                        f"UUID '{val}' not found in library.",
                        separator=self._separator
                    )
                    del self._lookup[key]

            # 4. Persist the cleaned lookup back to the JSON file
            try:
                with open(self._override_path, 'w') as f:
                    # noinspection PyTypeChecker
                    json.dump(self._lookup, f, indent=4)
                self._logger.trace(
                    f"Wrote updated manual overrides ({len(self._lookup)} entries).",
                    separator=self._separator
                )
            except Exception as e:
                self._logger.error(
                    f"Failed to write cleaned overrides to {self._override_path}: {e}",
                    separator=self._separator
                )

        # 5. Store the (possibly updated) lookup in the context
        ctx.manual_override_lookup = self._lookup
        return ctx

    @staticmethod
    def _get_missing_uuids(lib_df: pd.DataFrame, uuids: List[str]) -> List[str]:
        missing = pd.Index(uuids).difference(lib_df["UUID"])
        return missing.tolist()

    def _build_lookup(self) -> Dict[str, str]:
        if not self._override_path.is_file():
            self._logger.warning(
                "Cannot load override cache, file is non-existent.",
                separator=self._separator
            )
            return {}
        if not self._override_path.name.endswith('.json'):
            self._logger.warning(
                "Cannot load override cache, file is not json.",
                separator=self._separator
            )
            return {}

        with open(self._override_path, 'r') as f:
            data = json.load(f)
        return data

