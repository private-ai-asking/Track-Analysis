from pathlib import Path
from typing import List, Tuple
import traceback

import mutagen
import numpy as np
import pandas as pd

from track_analysis.components.md_common_python.py_common.logging import HoornLogger
from track_analysis.components.track_analysis.features.data_generation.util.key_extractor import (
    KeyExtractor,
    KeyExtractionResult,
)


class KeyTaggingService:
    def __init__(self, extractor: KeyExtractor, csv_path: Path, logger: HoornLogger):
        self._extractor = extractor
        self._csv_path = csv_path
        self._logger = logger

    def tag_and_record(
            self,
            paths: List[Path]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[KeyExtractionResult]]:
        results = self._safe_extract(paths)
        gk, sk, ek, rows = self._process(results, paths)
        self._append(rows)
        return gk, sk, ek, results

    def _safe_extract(self, paths: List[Path]) -> List[KeyExtractionResult]:
        try:
            # KeyExtractor expects a list of (index, Path) tuples
            return self._extractor.extract_keys_batch(list(enumerate(paths)))
        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error(f"KeyExtractor failed: {e}\n{tb}", separator=self.__class__.__name__)
            return []

    # noinspection t
    def _process(
            self,
            results: List[KeyExtractionResult],
            paths: List[Path]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        count = len(paths)
        global_keys = np.full(count, "", dtype=object)
        start_keys  = np.full(count, "", dtype=object)
        end_keys    = np.full(count, "", dtype=object)
        rows: List[dict] = []

        for res in results:
            idx = res.index
            global_keys[idx] = res.global_key

            path = paths[idx]
            # write tags into FLAC files
            if path.suffix.lower() == ".flac":
                self._write_flac_key_tag(path, res.global_key)

            if res.local_info:
                # first/last local key
                start_keys[idx] = res.local_info[0].key
                end_keys[idx]   = res.local_info[-1].key

                # build CSV rows for each segment
                for seg in res.local_info:
                    rows.append({
                        "File Path":      str(path),
                        "Segment Start":  seg.interval_start,
                        "Segment End":    seg.interval_end,
                        "Segment Key":    seg.key,
                    })

        return global_keys, start_keys, end_keys, rows

    def _append(self, rows: List[dict]) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        df.to_csv(
            self._csv_path,
            mode="a",
            index=False,
            header=not self._csv_path.exists()
        )

    def _write_flac_key_tag(self, path: Path, key: str) -> None:
        try:
            tag_file = mutagen.File(str(path), easy=True)
            if tag_file is None:
                return
            tag_file["initialkey"]  = key
            tag_file["global_key"]  = key
            tag_file.save()
        except Exception:
            # best effort; failures here shouldn't stop the pipeline
            pass
