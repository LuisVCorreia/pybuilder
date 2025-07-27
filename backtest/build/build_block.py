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
        block_data = storage.read_block_data(args.block)
        storage.close()

        if args.only_order_ids:
            block_data.filter_orders_by_ids(args.only_order_ids)

        block_data.filter_late_orders(args.block_building_time_ms)

        print(f"Block: {block_data.block_number} {block_data.onchain_block.get('hash')}")
        print(f"bid value: {block_data.winning_bid_trace.get('value')}")
        print(f"builder pubkey: {block_data.winning_bid_trace.get('builder_pubkey')}")

        return block_data

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

    context = SimulationContext.from_onchain_block(
        order_source.block_data.onchain_block,
        order_source.block_data.winning_bid_trace
    )

    logger.info("Creating EVM simulator...")
    evm_simulator = EVMSimulator(context, rpc_url)

    logger.info("Simulating orders...")
    simulated_orders = simulate_orders(order_objects, evm_simulator)
    logger.info(f"Simulation complete. Got {len(simulated_orders)} simulated orders.")

    # Print simulation results
    successful_sims = [sim for sim in simulated_orders if sim.simulation_result.success]
    failed_sims = [sim for sim in simulated_orders if not sim.simulation_result.success]

    # Save successful simulation state traces to file
    with open("state_trace_pybuilder.txt", "a") as f:
        for sim in successful_sims:
            f.write(f"Order {sim.order.id()} state trace:\n")
            f.write(sim.simulation_result.state_trace.summary() + "\n")

    logger.info(f"Successful simulations: {len(successful_sims)}")
    logger.info(f"Failed simulations: {len(failed_sims)}")

    with open("pybuilder_sim_results.txt", "w") as f:
        for sim in simulated_orders:
            f.write(f"Simulating order: {sim.order.id()}\n")
            f.write(f"Simulation result: true, "
                    f"gas used: {sim.sim_value.gas_used}, "
                    f"coinbase profit: {sim.sim_value.coinbase_profit}\n")

    if successful_sims:
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

    # Run builders
    logger.info("Running builders...")
    
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
    
    logger.info(f"Running builders: {builder_names}")
    
    # Run all builders
    results = run_builders(successful_sims, config, builder_names, evm_simulator)
    
    # Display comparison
    BuilderComparison.print_comparison(results)
    
    # Display detailed winning builder results in rbuilder style
    BuilderComparison.print_winning_builder_details(results)
    
    # Select and log winner
    best_result = BuilderComparison.select_best_builder(results)
    if best_result:
        logger.info(f"Block building complete. Best builder: {best_result.builder_name} "
                   f"with {best_result.bid_value / 10**18:.6f} ETH")
    else:
        logger.info("Block building complete. No successful builders.")
