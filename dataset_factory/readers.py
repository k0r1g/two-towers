from pathlib import Path
import pandas as pd

RAW_PARQUET_DIR = Path("data") / "raw" / "parquet"

def load_split(split: str = "train") -> pd.DataFrame:
    """Return the MS-MARCO split as a pandas DataFrame."""
    file_path = RAW_PARQUET_DIR / f"{split}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} not found. Run your download script first.")
    return pd.read_parquet(file_path) 