import logging
import os
from backtest.common.store import HistoricalDataStorage
from backtest.build.simulation.evm_simulator import simulate_orders, EVMSimulator
from backtest.build.builders.ordering_builder import run_builders
from backtest.build.builders.block_result import BuilderComparison
from backtest.build.simulation.sim_utils import SimulationContext
logger = logging.getLogger(__name__)

class LandedBlockFromDBOrdersSource:
    def __init__(self, args, config):
        self.config = config
        self.block_data = self.read_block_data(args, config)

    def read_block_data(self, args, config):
        storage = HistoricalDataStorage(config["fetch_sqlite_db_path"])
        try:
            block_data = storage.read_block_data(args.block)
        except Exception as e:
            logger.error(f"Error reading block data: {e}")
            block_data = None
        finally:
            storage.close()

        if not block_data:
            return None

        if args.only_order_ids:
            block_data.filter_orders_by_ids(args.only_order_ids)

        block_data.filter_late_orders(args.block_building_time_ms)

        print(f"Block: {block_data.block_number} {block_data.onchain_block.get('hash')}")
        print(f"bid value: {block_data.winning_bid_trace.get('value')}")
        print(f"builder pubkey: {block_data.winning_bid_trace.get('builder_pubkey')}")

        return block_data
    
def show_sim_results(successful_sims, failed_sims):
    if successful_sims:
        print("Successful simulations:")
        for sim in successful_sims:
            print(f"Order ID: {sim.order.id().value} | "
                    f"Profit: {sim.sim_value.coinbase_profit / 10**18:.18f} ETH | "
                    f"Gas Used: {sim.sim_value.gas_used:>7,} | "
                    f"Blob Gas Used: {sim.sim_value.blob_gas_used:,}")

        total_profit = sum(sim.sim_value.coinbase_profit for sim in successful_sims)
        total_gas = sum(sim.sim_value.gas_used for sim in successful_sims)
        total_blob_gas = sum(sim.sim_value.blob_gas_used for sim in successful_sims)

    if failed_sims:
        print("Failed simulations:")
        for sim in failed_sims:
            print(f"Order ID: {sim.order.id().value} | "
                    f"Error: {sim.simulation_result.error_message}")

    print(f"Total simulated profit: {total_profit / 10**18:.18f} ETH")
    print(f"Total simulated gas used: {total_gas:,}")
    print(f"Total blob gas used: {total_blob_gas:,}")


def run_backtest(args, config):
    logger.info("Starting backtest block build...")
    order_source = LandedBlockFromDBOrdersSource(args, config)
    if not order_source.block_data: return
    
    orders = order_source.block_data.available_orders
    logger.info(f"Got {len(orders)} orders to process")

    # Extract just the Order objects for simulation
    order_objects = [order_with_ts.order for order_with_ts in orders]

    rpc_url = config.get('fetch_rpc_url')
    if not rpc_url:
        raise ValueError("fetch_rpc_url not found in config")
    
    # Expand environment variables in the URL
    rpc_url = os.path.expandvars(rpc_url)

    context = SimulationContext.from_onchain_block(
        order_source.block_data.onchain_block,
        order_source.block_data.winning_bid_trace
    )

    logger.info("Creating EVM simulator...")
    evm_simulator = EVMSimulator(context, rpc_url)

    logger.info("Simulating orders...")
    simulated_orders = simulate_orders(order_objects, evm_simulator)
    logger.info(f"Simulation complete. Got {len(simulated_orders)} simulated orders.")

    successful_sims = [sim for sim in simulated_orders if sim.simulation_result.success]

    if args.show_sim:
        failed_sims = [sim for sim in simulated_orders if not sim.simulation_result.success]
        show_sim_results(successful_sims, failed_sims)

    if args.no_block_building:
        logger.info("Skipping block building.")
        return

    # Get builder names from args or config
    builder_names = getattr(args, 'builders', None)
    if not builder_names:
        # Default to all ordering builders in config
        builder_names = [
            b['name'] for b in config.get('builders', []) 
            if b.get('algo') == 'ordering-builder'
        ]
    
    if not builder_names:
        logger.warning("No builders specified or found in config")
        return
    
    logger.info(f"Running builders: {', '.join(builder_names)}")
    
    # Run all builders
    results = run_builders(successful_sims, config, args, builder_names, evm_simulator)

    if not results:
        return

    # Display comparison
    BuilderComparison.print_comparison(results)
    
    # Display detailed winning builder results in rbuilder style
    BuilderComparison.print_winning_builder_details(results)
    
    # Select and log winner
    best_result = BuilderComparison.select_best_builder(results)
    if best_result:
        logger.info(f"Block building complete. Best builder: {best_result.builder_name} "
                   f"with {best_result.bid_value / 10**18:.18f} ETH")
    else:
        logger.info("Block building complete. No successful builders.")
