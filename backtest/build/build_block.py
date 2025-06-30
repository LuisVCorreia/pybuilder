import logging
from backtest.common.store import HistoricalDataStorage

logger = logging.getLogger(__name__)

class LandedBlockFromDBOrdersSource:
    def __init__(self, args, config):
        self.config = config
        self.block_data = self.read_block_data(args, config)

    def read_block_data(self, args, config):
        storage = HistoricalDataStorage(config["fetch_sqlite_db_path"])
        block_data = storage.read_block_data(args.block)
        storage.close()

        if args.only_order_ids:
            block_data.filter_orders_by_ids(args.only_order_ids)

        if args.block_building_time_ms > 0:
            block_data.filter_late_orders(args.block_building_time_ms)

        if args.show_missing:
            self.show_missing_txs(block_data)

        print(f"Block: {block_data.block_number} {block_data.onchain_block.get('hash')}")
        print(f"bid value: {block_data.winning_bid_trace.get('value')}")
        print(f"builder pubkey: {block_data.winning_bid_trace.get('builder_pubkey')}")

        return block_data

    def show_missing_txs(self, block_data):
        missing_txs = block_data.search_missing_txs_on_available_orders()
        if missing_txs:
            logger.info(f"{len(missing_txs)} of txs by hashes missing on available orders")
            for missing_tx in missing_txs:
                logger.info(f"Tx: {missing_tx}")
        
        missing_nonce_txs = block_data.search_missing_account_nonce_on_available_orders()
        if missing_nonce_txs:
            logger.info(f"\n{len(missing_nonce_txs)} of txs by nonce pairs missing on available orders")
            for missing_nonce_tx in missing_nonce_txs:
                logger.info(f"Tx: {missing_nonce_tx[0]}, Account: {missing_nonce_tx[1]['account']}, Nonce: {missing_nonce_tx[1]['nonce']}")

def run_backtest_build_block(args, config, order_source):
    """Simulates and builds a block from a given order source."""
    orders = order_source.block_data.available_orders
    logger.info(f"Got {len(orders)} orders to process")

    # Placeholder for simulation logic
    logger.info("Simulating orders...")
    # sim_results = simulate_orders(orders, config)
    # logger.info(f"Simulation complete. Got {len(sim_results)} successful simulations.")

    # Placeholder for builder logic
    logger.info("Running builders...")
    # best_bid = run_builders(sim_results, config, args.builders)
    # logger.info(f"Block building complete. Best bid: {best_bid}")

def run_backtest(args, config):
    logger.info("Starting backtest block build...")
    order_source = LandedBlockFromDBOrdersSource(args, config)
    run_backtest_build_block(args, config, order_source)
