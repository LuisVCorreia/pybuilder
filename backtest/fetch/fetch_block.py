from datetime import datetime, timedelta, timezone
from ..common.root_provider import RootProvider
from ..common.mempool import ensure_parquet_files
from ..common.order import filter_orders_by_base_fee, fetch_transactions
from ..common.store import init_db, write_block_data
from ..common.mev_boost import fetch_winning_bid_trace
import logging

sec_to_ms = lambda s: int(s * 1000)
window_before_sec = 180 # 3 mins
window_after_sec = 5

logger = logging.getLogger(__name__)

def fetch_and_store_block(
    block_number: int,
    provider_url: str,
    mempool_data_dir: str,
    sqlite_db_path: str,
):
    """
    Fetch a block and its mempool txs, and store in SQLite DB.
    Args:
        block_number: Block number to fetch
        provider_url: Ethereum node RPC URL
        mempool_data_dir: Directory for mempool Parquet files
        sqlite_db_path: Path to SQLite DB
    """

    # Fetch block from provider
    provider = RootProvider(provider_url)
    block = provider.get_block(block_number, full_transactions=True)
    block_ts = block['timestamp']  # In seconds
    block_ts_ms = block_ts * 1_000
    from_ts_ms  = block_ts_ms - (window_before_sec * 1_000)
    to_ts_ms    = block_ts_ms + (window_after_sec  * 1_000)

    block_dt = datetime.fromtimestamp(block_ts, tz=timezone.utc)
    from_dt = block_dt - timedelta(seconds=window_before_sec)
    to_dt   = block_dt + timedelta(seconds=window_after_sec)

    parquet_files = ensure_parquet_files(mempool_data_dir, from_dt, to_dt)
    mempool_txs = fetch_transactions(parquet_files, from_ts_ms, to_ts_ms)

    logger.info("Fetched orders, unfiltered: %d", len(mempool_txs))

    base_fee = block["baseFeePerGas"]
    mempool_txs = filter_orders_by_base_fee(base_fee, mempool_txs)

    logger.info("Filtered orders by base fee: %d", len(mempool_txs))

    block_hash = block["hash"].hex() if hasattr(block["hash"], "hex") else str(block["hash"])
    
    try:
        bid_trace = fetch_winning_bid_trace(block_hash, block_number)
    except RuntimeError as e:
        logger.warning(f"Skipping block {block_number}: {e}")
        return  # nothing more to do for this block

    # Store block and orders in SQLite
    conn = init_db(sqlite_db_path)
    write_block_data(conn, block_number, bid_trace, mempool_txs)
    conn.close()
    logger.info(f"Block {block_number} and {len(mempool_txs)} mempool txs stored in {sqlite_db_path}")
