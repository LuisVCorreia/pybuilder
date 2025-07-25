import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from collections import defaultdict

from backtest.build.simulation.sim_utils import SimulatedOrder, SimValue
from backtest.build.simulation.evm_simulator import EVMSimulator
from backtest.common.order import OrderId, TxNonce
from .order_priority import (
    Sorting, 
    PrioritizedOrderStore, 
    create_priority_class,
    MIN_SIM_RESULT_PERCENTAGE
)
from .block_result import BlockResult
from .block_building_helper import BlockBuildingHelper, ExecutionError

logger = logging.getLogger(__name__)

@dataclass
class OrderingBuilderConfig:
    """
    Configuration for the ordering builder.
    """
    discard_txs: bool = True
    sorting: Sorting = Sorting.MAX_PROFIT
    failed_order_retries: int = 1
    drop_failed_orders: bool = True
    coinbase_payment: bool = False
    build_duration_deadline_ms: Optional[int] = None

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'OrderingBuilderConfig':
        """Create config from dictionary (parsed from YAML)."""
        return cls(
            discard_txs=config_dict.get('discard_txs', True),
            sorting=Sorting.from_str(config_dict.get('sorting', 'max-profit')),
            failed_order_retries=config_dict.get('failed_order_retries', 1),
            drop_failed_orders=config_dict.get('drop_failed_orders', True),
            coinbase_payment=config_dict.get('coinbase_payment', False),
            build_duration_deadline_ms=config_dict.get('build_duration_deadline_ms')
        )

    def build_duration_deadline(self) -> Optional[float]:
        """Get build duration deadline in seconds."""
        if self.build_duration_deadline_ms is None:
            return None
        return self.build_duration_deadline_ms / 1000.0

class OrderingBuilder:
    """
    Main ordering builder implementation.
    The algorithm continuously processes orders from highest to lowest priority,
    attempting to include them in the block while handling execution failures
    and nonce conflicts.
    """
    
    def __init__(self, config: OrderingBuilderConfig, name: str):
        self.config = config
        self.name = name
        self.priority_class = create_priority_class(config.sorting)
        logger.debug(f"Created {name} with sorting: {config.sorting.value}")
    
    def build_block(
        self,
        simulated_orders: List[SimulatedOrder], 
        evm_simulator: EVMSimulator
    ) -> BlockResult:
        """
        Build a block using the ordering algorithm with in-block EVM simulation.

        Args:
            simulated_orders: List of successfully simulated orders
            evm_simulator: EVM simulator instance for in-block execution
            
        Returns:
            Block building result with included orders and metrics
        """
        start_time = time.time()
        
        try:
            initial_nonces = self._extract_initial_nonces(simulated_orders)
            
            order_store = PrioritizedOrderStore(self.priority_class, initial_nonces)
            failed_orders: Set[OrderId] = set()
            order_attempts: Dict[OrderId, int] = defaultdict(int)
            
            # Insert all valid orders into the priority store
            valid_orders = self._filter_valid_orders(simulated_orders)
            for order in valid_orders:
                order_store.insert_order(order)
            
            logger.info(
                f"Building block with {len(valid_orders)} orders "
                f"({len(simulated_orders) - len(valid_orders)} filtered) using {self.name}"
            )
            
            # Initialize block building helper with the shared simulator
            helper = BlockBuildingHelper(self.name, evm_simulator)

            evm_simulator.fork_at_block(evm_simulator.context.block_number - 1)

            # Main block building loop
            self._fill_orders(order_store, helper, failed_orders, order_attempts)
            
            # Finalize and return result
            result = helper.finalize_block()
            result.build_time_ms = (time.time() - start_time) * 1000
            
            logger.info(
                f"{self.name} built block: {result.bid_value / 10**18:.6f} ETH, "
                f"{result.total_gas_used:,} gas, {len(result.included_orders)} orders "
                f"in {result.build_time_ms:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            build_time_ms = (time.time() - start_time) * 1000
            logger.error(f"{self.name} block building failed: {e}")
            return BlockResult(
                builder_name=self.name,
                success=False,
                error_message=str(e),
                build_time_ms=build_time_ms
            )
    
    def _extract_initial_nonces(self, simulated_orders: List[SimulatedOrder]) -> List[TxNonce]:
        """
        Extract the initial nonce state from the orders themselves.
        
        For backtesting, we assume that the minimum nonce seen for each account
        represents the current on-chain nonce at the start of block building.
        This means that transaction with the minimum nonce is ready to execute.
        
        Args:
            simulated_orders: List of simulated orders
            
        Returns:
            List of TxNonce representing initial nonce state
        """
        account_nonces = {}
        
        # Find the minimum nonce for each account across all orders
        for order in simulated_orders:
            for nonce_info in order.order.nonces():
                account = nonce_info.address
                nonce = nonce_info.nonce
                
                if account not in account_nonces or nonce < account_nonces[account]:
                    account_nonces[account] = nonce
        
        # Convert to TxNonce list
        # The minimum nonce for each account represents the current on-chain nonce
        initial_nonces = []
        for account, nonce in account_nonces.items():
            initial_nonces.append(TxNonce(address=account, nonce=nonce, optional=False))
        
        return initial_nonces

    def _filter_valid_orders(self, simulated_orders: List[SimulatedOrder]) -> List[SimulatedOrder]:
        """
        Filter out invalid orders before processing.
        """
        valid_orders = []
        for order in simulated_orders:
            # Skip orders with 0 gas (placeholder orders)
            if order.sim_value.gas_used == 0:
                logger.debug(f"Skipping order {order.order.id()} with 0 gas")
                continue
            
            # Skip orders with negative profit in profit-based sorting
            if (self.config.sorting in [Sorting.MAX_PROFIT] and 
                order.sim_value.coinbase_profit <= 0):
                logger.debug(f"Skipping order {order.order.id()} with non-positive profit")
                continue
            
            valid_orders.append(order)
        
        return valid_orders
    
    def _fill_orders(
        self, 
        order_store: PrioritizedOrderStore, 
        helper: BlockBuildingHelper,
        failed_orders: Set[OrderId],
        order_attempts: Dict[OrderId, int]
    ) -> None:
        """
        Fill orders into the block following the ordering algorithm.

        Args:
            order_store: Priority queue of orders
            helper: Block building helper for execution
            failed_orders: Set of permanently failed order IDs
            order_attempts: Retry count per order
        """
        build_start = time.time()
        orders_processed = 0
        
        while not order_store.is_empty():
            if self.config.build_duration_deadline_ms:
                elapsed_ms = (time.time() - build_start) * 1000
                if elapsed_ms > self.config.build_duration_deadline_ms:
                    logger.debug(f"Build deadline reached: {elapsed_ms:.2f}ms")
                    break
            
            # Get next highest priority order
            sim_order = order_store.pop_order()
            if not sim_order:
                break
            
            order_id = sim_order.order.id()
            orders_processed += 1
            
            # Skip already failed orders
            if order_id in failed_orders:
                continue
            
            # Check if we can fit this order
            if not helper.can_add_order(sim_order):
                logger.debug(f"Order {order_id} would exceed gas limit, skipping")
                continue
            
            logger.debug(f"Attempting to include order {order_id} (attempt {order_attempts[order_id] + 1})")

            def profit_validator(order_id: OrderId, original_sim: SimValue, new_sim: SimValue) -> None:
                mode = self.config.sorting.value
                if self.priority_class.simulation_too_low(original_sim, new_sim):
                    if mode == Sorting.MAX_PROFIT.value:
                        before, after = original_sim.coinbase_profit, new_sim.coinbase_profit
                        metric = "profit"
                    else:
                        before, after = original_sim.mev_gas_price, new_sim.mev_gas_price
                        metric = "MEV gas price"

                    threshold = (before * MIN_SIM_RESULT_PERCENTAGE) // 100

                    raise ExecutionError(
                        f"[{mode}] Order {order_id} {metric} dropped "
                        f"{after} < {threshold} ({MIN_SIM_RESULT_PERCENTAGE}%)",
                        error_type="sim_too_low",
                        new_sim=new_sim
                    )

            
            # Attempt to commit the order
            try:
                new_sim_value, nonces_updates = helper.commit_order(sim_order, profit_validator)
                
                # Order succeeded - update nonces in order store
                nonces_updated = []
                for account, nonce in nonces_updates:
                    nonces_updated.append(TxNonce(address=account, nonce=nonce, optional=False))
                
                if nonces_updated:
                    logger.debug(f"Updating nonces after order {order_id}:")
                    for nonce_update in nonces_updated:
                        logger.debug(f"  {nonce_update.address}: -> {nonce_update.nonce}")
                    order_store.update_onchain_nonces(nonces_updated)
                
                logger.debug(
                    f"Included order {order_id}: {new_sim_value.gas_used:,} gas, "
                    f"{new_sim_value.coinbase_profit} wei profit"
                )
                
            except ExecutionError as e:
                # Order failed - handle retry logic
                self._handle_order_failure(
                    sim_order, e, order_store, failed_orders, order_attempts
                )
        
        logger.debug(f"Processed {orders_processed} orders, included {len(helper.included_orders)}")
    
    def _handle_order_failure(
        self,
        sim_order: SimulatedOrder,
        error: ExecutionError,
        order_store: PrioritizedOrderStore,
        failed_orders: Set[OrderId],
        order_attempts: Dict[OrderId, int]
    ) -> None:
        """
        Handle order execution failure with retry logic.
        
        Args:
            sim_order: The failed order
            error: The execution error
            order_store: Priority store for retries
            failed_orders: Set of permanently failed orders
            order_attempts: Retry counts per order
        """
        order_id = sim_order.order.id()
        attempt_count = order_attempts[order_id]
        
        # Check if we can retry with lower expected profit
        if (attempt_count < self.config.failed_order_retries and 
            error.error_type == "sim_too_low"):

            if error.new_sim is not None:
                # Update simulation value with new lower expected profit
                sim_order.sim_value = error.new_sim

            # Re-insert order with updated simulation value for retry
            # In a real implementation, this would use the actual execution result
            logger.info(f"Retrying order {order_id} (attempt {attempt_count + 1}/{self.config.failed_order_retries})")
            logger.info(error)
            order_attempts[order_id] += 1
            order_store.insert_order(sim_order)
            
        else:
            # Mark order as permanently failed
            if self.config.drop_failed_orders:
                failed_orders.add(order_id)
            
            logger.info(
                f"Order {order_id} failed permanently: {error} "
                f"(attempts: {attempt_count + 1})"
            )


def create_ordering_builder(builder_config: dict) -> OrderingBuilder:
    """
    Factory function to create an ordering builder from configuration.
    
    Args:
        builder_config: Builder configuration dictionary
        
    Returns:
        Configured OrderingBuilder instance
    """
    config = OrderingBuilderConfig.from_dict(builder_config)
    name = builder_config.get('name', 'ordering-builder')
    return OrderingBuilder(config, name)


def run_builders(
    simulated_orders: List[SimulatedOrder], 
    config: dict, 
    builder_names: List[str],
    evm_simulator: EVMSimulator
) -> List[BlockResult]:
    """
    Run multiple builders and return their results with in-block EVM simulation.
    
    Args:
        simulated_orders: Successfully simulated orders
        config: Full configuration dictionary
        builder_names: Names of builders to run
        evm_simulator: Pre-created EVM simulator instance for reuse
        
    Returns:
        List of block building results
    """
    results = []
    
    # Get builder configurations
    builder_configs = {b['name']: b for b in config.get('builders', [])}
    
    for builder_name in builder_names:
        if builder_name not in builder_configs:
            logger.error(f"Builder {builder_name} not found in config")
            results.append(BlockResult(
                builder_name=builder_name,
                success=False,
                error_message=f"Builder {builder_name} not found in config"
            ))
            continue
        
        builder_config = builder_configs[builder_name]
        builder_algo = builder_config.get('algo')
        
        if builder_algo == 'ordering-builder':
            builder = create_ordering_builder(builder_config)
            result = builder.build_block(simulated_orders, evm_simulator)
            results.append(result)
            
        elif builder_algo == 'parallel-builder':
            from .parallel_builder import run_parallel_builder
            result = run_parallel_builder(simulated_orders, config, evm_simulator)
            results.append(result)
            
        else:
            logger.warning(f"Unknown builder algorithm: {builder_algo} for {builder_name}")
            results.append(BlockResult(
                builder_name=builder_name,
                success=False,
                error_message=f"Unknown builder algorithm: {builder_algo}"
            ))
    
    return results
