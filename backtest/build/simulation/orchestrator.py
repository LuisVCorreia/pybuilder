import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from backtest.common.order import Order
from backtest.common.block_data import BlockData
from .state_provider import StateProviderFactory, SimulationContext
from .mock_provider import MockStateProviderFactory
from .rpc_provider import AlchemyStateProviderFactory
from .simulator import SimpleOrderSimulator, SimulatedOrder
from .evm_simulator import EVMSimulator

logger = logging.getLogger(__name__)


class StateProviderType(Enum):
    """Types of state providers"""
    MOCK = "mock"
    ALCHEMY = "alchemy"

@dataclass
class SimulationConfig:
    """Configuration for simulation"""
    # State provider configuration
    state_provider_type: StateProviderType
    state_provider_config: Dict[str, Any]
    
    # Simulation options
    simulation_timeout: int = 30
    cache_state: bool = True
    
    # Logging options
    log_level: str = "INFO"
    log_simulation_details: bool = False


class SimulationOrchestrator:
    """Orchestrates order simulation with configurable state providers"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self._setup_logging()
        self.state_provider_factory = self._create_state_provider_factory()
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.getLogger("backtest.simulation").setLevel(log_level)
    
    def _create_state_provider_factory(self) -> StateProviderFactory:
        """Create state provider factory based on configuration"""
        provider_type = self.config.state_provider_type
        provider_config = self.config.state_provider_config
        
        if provider_type == StateProviderType.ALCHEMY:
            rpc_url = provider_config.get("rpc_url")
            if not rpc_url:
                raise ValueError("Alchemy state provider requires 'rpc_url' in config")
            return AlchemyStateProviderFactory(rpc_url)
        
        else:
            raise ValueError(f"Unsupported state provider type: {provider_type}")
    
    def simulate_orders(self, orders: List[Order], block_data: BlockData) -> List[SimulatedOrder]:
        """
        Simulate orders using onchain block data for proper context.
        This is the preferred method that mirrors rbuilder's approach.
        
        Args:
            orders: List of orders to simulate
            block_data: Block data containing onchain block and winning bid trace
            
        Returns:
            List of simulated orders with results
        """
        try:
            context = SimulationContext.from_onchain_block(block_data.onchain_block)
            parent_block_number = context.block_number - 1
            parent_state_provider = self.state_provider_factory.history_by_block_number(parent_block_number)
            logger.info(f"Simulating {len(orders)} orders for block {context.block_number}")
            logger.info(f"Using onchain block data: hash={context.block_hash}")
            if self.config.state_provider_type == StateProviderType.ALCHEMY:
                simulator = EVMSimulator(parent_state_provider, context)
                logger.info(f"Using enhanced EVM-based simulation with onchain block context (parent state root)")
                results = []
                for order in orders:
                    result = simulator.simulate_order(order)
                    results.append(result)
            else:
                simulator = SimpleOrderSimulator(self.state_provider_factory)
                results = simulator.simulate_orders(orders, context)
                logger.info(f"Using simple validation-based simulation")
            
            # Log summary
            successful = sum(1 for r in results if r.simulation_result.success)
            failed = len(results) - successful
            total_gas = sum(r.simulation_result.gas_used for r in results if r.simulation_result.success)
            total_profit = sum(r.simulation_result.coinbase_profit for r in results if r.simulation_result.success)
            
            logger.info(f"Block {context.block_number} simulation completed: {successful} successful, {failed} failed")
            logger.info(f"Total gas used: {total_gas}, Total coinbase profit: {total_profit} wei")
            
            return results
            
        except Exception as e:
            logger.error(f"Block simulation failed: {e}")
            raise
    