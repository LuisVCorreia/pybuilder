import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from backtest.common.order import Order
from .state_provider import StateProviderFactory
from .mock_provider import MockStateProviderFactory
from .simulator import SimpleOrderSimulator, SimulatedOrder

logger = logging.getLogger(__name__)


class StateProviderType(Enum):
    """Types of state providers"""
    MOCK = "mock"
    # ALCHEMY = "alchemy"
    # IPC = "ipc"

@dataclass
class SimulationConfig:
    """Configuration for simulation"""
    # State provider configuration
    state_provider_type: StateProviderType
    state_provider_config: Dict[str, Any]
    
    # Simulation options
    enable_transaction_decoding: bool = True
    enable_bundle_simulation: bool = True
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
        
        if provider_type == StateProviderType.MOCK:
            block_number = provider_config.get("block_number", 18000000)
            return MockStateProviderFactory(block_number)
        
        elif provider_type == StateProviderType.ALCHEMY:
            raise ValueError("Alchemy state provider not available in build simulation")
        
        else:
            raise ValueError(f"Unsupported state provider type: {provider_type}")
    
    def simulate_orders(self, orders: List[Order], block_number: int) -> List[SimulatedOrder]:
        """
        Simulate a list of orders using the configured state provider
        
        Args:
            orders: List of orders to simulate
            block_number: Block number for simulation context
            
        Returns:
            List of simulated orders with results
        """
        try:
            # Get simulation context
            context = self.state_provider_factory.get_simulation_context(block_number)
            
            # Create simulator
            simulator = SimpleOrderSimulator(self.state_provider_factory)
            
            # Log simulation start
            if self.config.log_simulation_details:
                logger.info(f"Starting simulation of {len(orders)} orders at block {block_number}")
                logger.info(f"Block context: timestamp={context.block_timestamp}, base_fee={context.block_base_fee}")
            
            # Simulate orders
            results = simulator.simulate_orders(orders, context)
            
            # Log summary
            successful = sum(1 for r in results if r.simulation_result.success)
            failed = len(results) - successful
            total_gas = sum(r.simulation_result.gas_used for r in results if r.simulation_result.success)
            total_profit = sum(r.simulation_result.coinbase_profit for r in results if r.simulation_result.success)
            
            logger.info(f"Simulation completed: {successful} successful, {failed} failed")
            logger.info(f"Total gas used: {total_gas}, Total coinbase profit: {total_profit} wei")
            
            if self.config.log_simulation_details:
                for i, result in enumerate(results):
                    sr = result.simulation_result
                    logger.debug(f"Order {i}: {sr.success}, gas={sr.gas_used}, profit={sr.coinbase_profit}")
                    if not sr.success:
                        logger.debug(f"  Error: {sr.error}, {sr.error_message}")
            
            return results
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise
    
    def simulate_single_order(self, order: Order, block_number: int) -> SimulatedOrder:
        """Simulate a single order"""
        results = self.simulate_orders([order], block_number)
        return results[0] if results else None
