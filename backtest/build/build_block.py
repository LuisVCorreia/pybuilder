import logging
import os
from backtest.common.store import HistoricalDataStorage
from backtest.build.simulation.evm_simulator import simulate_orders

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

def run_backtest(args, config):
    logger.info("Starting backtest block build...")
    order_source = LandedBlockFromDBOrdersSource(args, config)
    
    orders = order_source.block_data.available_orders
    logger.info(f"Got {len(orders)} orders to process")

    # Extract just the Order objects for simulation
    order_objects = [order_with_ts.order for order_with_ts in orders]

    rpc_url = config.get('fetch_rpc_url')
    if not rpc_url:
        raise ValueError("fetch_rpc_url not found in config")
    
    # Expand environment variables in the URL
    rpc_url = os.path.expandvars(rpc_url)

    logger.info("Simulating orders...")
    simulated_orders = simulate_orders(order_objects, order_source.block_data, rpc_url)
    logger.info(f"Simulation complete. Got {len(simulated_orders)} simulated orders.")
    
    # Print simulation results
    successful_sims = [sim for sim in simulated_orders if sim.simulation_result.success]
    failed_sims = [sim for sim in simulated_orders if not sim.simulation_result.success]
    
    logger.info(f"Successful simulations: {len(successful_sims)}")
    logger.info(f"Failed simulations: {len(failed_sims)}")

    if successful_sims:
        # Use sim_value for successful orders (preferred way)
        total_profit = sum(sim.sim_value.coinbase_profit for sim in successful_sims)
        total_gas = sum(sim.sim_value.gas_used for sim in successful_sims)
        total_blob_gas = sum(sim.sim_value.blob_gas_used for sim in successful_sims)
        total_kickbacks = sum(sim.sim_value.paid_kickbacks for sim in successful_sims)
        
        logger.info(f"Total simulated profit: {total_profit / 10**18:.6f} ETH")
        logger.info(f"Total simulated gas used: {total_gas:,}")
        if total_blob_gas > 0:
            logger.info(f"Total blob gas used: {total_blob_gas:,}")
        if total_kickbacks > 0:
            logger.info(f"Total kickbacks paid: {total_kickbacks / 10**18:.6f} ETH")
    
    # Show failed simulations for debugging
    if failed_sims:
        logger.info("Failed simulations:")
        for sim in failed_sims:
            error_info = sim.simulation_result.error if sim.simulation_result.error else "UNKNOWN"
            error_msg = sim.simulation_result.error_message or "No error message"
            logger.info(f"  Order {sim.order.id()}: {error_info} - {error_msg}")

    # Placeholder for builder logic
    logger.info("Running builders...")
    # best_bid = run_builders(simulated_orders, config, args.builders)
    # logger.info(f"Block building complete. Best bid: {best_bid}")
