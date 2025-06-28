import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from backtest.config import load_config
from backtest.fetch.fetch_block import fetch_historical_data
from backtest.common.store import HistoricalDataStorage
from dotenv import load_dotenv
import logging
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Fetch a block and mempool txs, store in SQLite DB.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--block", type=int, help="Block number to fetch")
    parser.add_argument("--provider", type=str, help="Ethereum node RPC URL")
    parser.add_argument("--mempool-dir", type=str, help="Directory for mempool Parquet files")
    parser.add_argument("--db", type=str, help="Path to SQLite DB")
    args = parser.parse_args()

    config = load_config(args.config)
    # Set up logging from config
    log_level = getattr(logging, config.get("logging_level", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logger = logging.getLogger(__name__)

    block_number = args.block if args.block is not None else config.get("block_number")
    provider_url = args.provider if args.provider else config["provider_url"]
    mempool_data_dir = args.mempool_dir if args.mempool_dir else config["mempool_data_dir"]
    sqlite_db_path = args.db if args.db else config["sqlite_db_path"]

    if block_number is None:
        raise ValueError("Block number must be specified via --block or in config.yaml")

    block_data = fetch_historical_data(
        block_number=block_number,
        provider_url=provider_url,
        mempool_data_dir=mempool_data_dir,
    )

    # Store block data using HistoricalDataStorage
    storage = HistoricalDataStorage(sqlite_db_path)
    storage.write_block_data(block_data)
    storage.close()

    logger.info(f"Block {block_number} and {len(block_data.available_orders)} mempool txs stored in {sqlite_db_path}")


if __name__ == "__main__":
    main()
