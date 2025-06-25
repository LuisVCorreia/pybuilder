import pandas as pd
from typing import List

def filter_transactions(
    parquet_files: List[str],
    from_ts_ms: int,
    to_ts_ms: int
) -> pd.DataFrame:
    """
    Read and filter transactions from Parquet files for the given timestamp window,
    mirroring the Rust implementation:
      - select only `timestamp`, `rawTx`, and `hash`
      - strict `>` and `<` bounds
      - sort by timestamp
    Returns a DataFrame with columns:
      - timestamp_ms: int
      - raw_tx: bytes
      - hash: str
    """
    parts = []

    for path in parquet_files:
        # 1) Load only the three columns we need
        df = pd.read_parquet(path, columns=["timestamp", "rawTx", "hash"])

        # 2) Rename for clarity
        df = df.rename(columns={
            "timestamp": "timestamp_dt",
            "rawTx": "raw_tx",
        })

        # 3) Convert datetime64[ns] to integer ms since epoch
        df["timestamp_ms"] = (
            df["timestamp_dt"].values
              .astype("datetime64[ms]")
              .astype("int64")
        )

        # 4) Strict interval filter
        mask = (df["timestamp_ms"] > from_ts_ms) & (df["timestamp_ms"] < to_ts_ms)
        sliced = df.loc[mask, ["timestamp_ms", "raw_tx", "hash"]]

        if not sliced.empty:
            parts.append(sliced)

    # 5) If no transactions match, return an empty DataFrame
    if not parts:
        return pd.DataFrame(columns=["timestamp_ms", "raw_tx", "hash"])

    # 6) Concatenate, sort, and reindex
    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values("timestamp_ms").reset_index(drop=True)
    return result
