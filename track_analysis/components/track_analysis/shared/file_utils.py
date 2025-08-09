from pathlib import Path
import os
import pandas as pd


class FileUtils:
    def get_size_bytes(self, path: Path) -> int:
        return os.path.getsize(path)

    def write_csv(self, df: pd.DataFrame, path: Path, append: bool = True) -> None:
        df.to_csv(
            path,
            mode='a' if append else 'w',
            index=False,
            header=not append
        )
