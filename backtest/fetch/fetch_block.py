from datetime import datetime, timedelta, timezone
from .root_provider import RootProvider
from ..common.mempool import ensure_parquet_files
from ..common.order import filter_orders_by_base_fee, fetch_transactions, filter_orders_by_nonces
from ..common.mev_boost import fetch_winning_bid_trace
from ..common.block_data import BlockData, OrdersWithTimestamp
import logging

sec_to_ms = lambda s: int(s * 1000)
window_before_sec = 180 # 3 mins
window_after_sec = 5

logger = logging.getLogger(__name__)

def fetch_historical_data(
    block_number: int,
    provider_url: str,
    mempool_data_dir: str,
    concurrency_limit: int
):
    """
    Fetch a block and its mempool txs, and store in SQLite DB.
    Args:
        block_number: Block number to fetch
        provider_url: Ethereum node RPC URL
        mempool_data_dir: Directory for mempool Parquet files
    """

    logger.info("Fetching historical data for block %s", block_number)

    # Fetch block from provider
    provider = RootProvider(provider_url)
    block = provider.get_block(block_number, full_transactions=True)

    block_hash = block["hash"].hex() if hasattr(block["hash"], "hex") else str(block["hash"])
    logger.info("Fetching payload delivered")
    try:
        bid_trace = fetch_winning_bid_trace(block_number)
    except RuntimeError as e:
        logger.warning(f"Skipping block {block_number}: {e}")
        return  # Nothing more to do for this block

    block_ts = block['timestamp']  # In seconds
    block_ts_ms = block_ts * 1_000
    from_ts_ms  = block_ts_ms - (window_before_sec * 1_000)
    to_ts_ms    = block_ts_ms + (window_after_sec  * 1_000)

    block_dt = datetime.fromtimestamp(block_ts, tz=timezone.utc)
    from_dt = block_dt - timedelta(seconds=window_before_sec)
    to_dt   = block_dt + timedelta(seconds=window_after_sec)

    logger.info("Fetching block from eth provider")

    parquet_files = ensure_parquet_files(mempool_data_dir, from_dt, to_dt)
    mempool_txs = fetch_transactions(parquet_files, from_ts_ms, to_ts_ms)
    logger.info("Fetched orders, unfiltered. Orders left: %d", len(mempool_txs))

    base_fee = block["baseFeePerGas"]
    mempool_txs = filter_orders_by_base_fee(base_fee, mempool_txs)
    logger.info("Filtered orders by base fee. Orders left: %d", len(mempool_txs))

    mempool_txs = filter_orders_by_nonces(provider, mempool_txs, block_number, concurrency_limit)
    logger.info("Filtered orders by nonces. Orders left: %d", len(mempool_txs))

    mempool_txs.sort(key=lambda o: o.timestamp_ms)

    # Convert block dict to proper format for storage
    onchain_block = {
        'number': block_number,
        'hash': block_hash,
        'timestamp': block_ts,
        'baseFeePerGas': base_fee,
        'miner': block.get('miner', ''),
        'transactions': block.get('transactions', []),
        # TODO: add more fields as needed
        **block
    }
    
    # Convert orders to OrdersWithTimestamp format
    orders_with_timestamp = [
        OrdersWithTimestamp(timestamp_ms=order.timestamp_ms, order=order)
        for order in mempool_txs
    ]
    
    return BlockData(
        block_number=block_number,
        winning_bid_trace=bid_trace,
        onchain_block=onchain_block,
        available_orders=orders_with_timestamp,
        filtered_orders={},
        built_block_data=None
    )
