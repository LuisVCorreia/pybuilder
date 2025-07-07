import logging
import os
from typing import List

from backtest.common.order import Order
from backtest.common.block_data import BlockData
from .orchestrator import SimulationOrchestrator, SimulationConfig, StateProviderType
from .evm_simulator import SimulatedOrder

logger = logging.getLogger(__name__)

def simulate_orders_with_alchemy_evm(orders: List[Order], block_data: BlockData, config: dict) -> List[SimulatedOrder]:
    """
    Function for simulating orders with Alchemy RPC, EVM execution, and proper onchain block context.
    This method mirrors rbuilder's approach.

    Args:
        orders: List of orders to simulate
        block_data: Block data containing onchain block and winning bid trace for proper context
        config: Configuration dictionary containing fetch_rpc_url
        
    Returns:
        List of simulated orders with EVM-based results using proper block context
    """
    # Get Alchemy RPC URL from config
    rpc_url = config.get('fetch_rpc_url')
    if not rpc_url:
        raise ValueError("fetch_rpc_url not found in config")
    
    # Expand environment variables in the URL
    rpc_url = os.path.expandvars(rpc_url)
        
    # Create simulation configuration
    sim_config = SimulationConfig(
        state_provider_type=StateProviderType.ALCHEMY,
        state_provider_config={'rpc_url': rpc_url},
        log_level="INFO",
        log_simulation_details=False
    )
    
    # Create orchestrator and simulate with proper block context
    orchestrator = SimulationOrchestrator(sim_config)
    results = orchestrator.simulate_orders(orders, block_data)
    
    logger.info(f"Enhanced EVM simulation completed for {len(orders)} orders using onchain block context")
    return results
