import os
import requests
from datetime import datetime, timedelta
from typing import List
from alive_progress import alive_bar

BASE_URL = "https://mempool-dumpster.flashbots.net/ethereum/mainnet"
def download_parquet_for_day(data_dir: str, day_str: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{day_str}.parquet")
    month = day_str[:7]
    url = f"{BASE_URL}/{month}/{day_str}.parquet"

    existing = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    head = requests.head(url); head.raise_for_status()
    total = int(head.headers.get("Content-Length", 0))
    if existing >= total:
        return file_path

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    resp = requests.get(url, stream=True, headers=headers)
    resp.raise_for_status()

    mode = "ab" if existing else "wb"
    with open(file_path, mode) as f, alive_bar(
        total, title=f"Downloading {day_str}.parquet", bar="blocks", spinner="dots_waves"
    ) as bar:
        bar(existing)
        for chunk in resp.iter_content(chunk_size=32_768):
            if not chunk:
                break
            f.write(chunk)
            bar(len(chunk))

    final = os.path.getsize(file_path)
    if final != total:
        raise IOError(f"Incomplete download: {final}/{total} bytes")
    return file_path


def ensure_parquet_files(data_dir: str, from_ts: datetime, to_ts: datetime) -> List[str]:
    """
    Ensure all Parquet files for the date range [from_ts, to_ts] exist locally.
    Args:
        data_dir: Directory to store Parquet files.
        from_ts: Start datetime.
        to_ts: End datetime.
    Returns:
        List of file paths for the required Parquet files.
    """
    current_day = from_ts.date()
    end_day = to_ts.date()
    files = []
    while current_day <= end_day:
        day_str = current_day.strftime("%Y-%m-%d")
        files.append(download_parquet_for_day(data_dir, day_str))
        current_day += timedelta(days=1)
    return files
