import hashlib
import pandas as pd

def get_dataframe_hash(df: pd.DataFrame) -> str:
    """
    Generates a stable SHA256 hash for a pandas DataFrame's content.
    """
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
