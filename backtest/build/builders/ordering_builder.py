import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Any, Callable
from collections import defaultdict
from .parallel_builder import run_parallel_builder

from backtest.build.simulation.sim_utils import SimulatedOrder, SimValue
from backtest.build.simulation.evm_simulator import EVMSimulator
from backtest.common.order import OrderId, TxNonce
from .order_priority import (
    Sorting, 
    PrioritizedOrderStore, 
    create_priority_class, 
)
from .block_result import BlockResult, BlockTrace

logger = logging.getLogger(__name__)

# Assume standard payout tx gas limit
# rbuilder uses more complex logic to estimate this, but for simple backtesting we can use a fixed value
PAYOUT_GAS_LIMIT = 21000  

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


class ExecutionError(Exception):
    """Exception raised when order execution fails."""
    def __init__(self, message: str, error_type: str = "execution_error"):
        super().__init__(message)
        self.error_type = error_type


class BlockBuildingHelper:
    """
    Helper class for building blocks with order execution.
    
    This class manages the incremental building of a block by tracking:
    - Gas usage and limits
    - Coinbase profit accumulation
    - Included orders and their execution
    - Nonce state updates
    """
    
    def __init__(self, builder_name: str, simulator: EVMSimulator):
        self.context = simulator.context
        self.builder_name = builder_name
        self.simulator = simulator  # Now required, not optional
        self.gas_used = 0
        self.coinbase_profit = 0
        self.blob_gas_used = 0
        self.paid_kickbacks = 0
        self.included_orders: List[SimulatedOrder] = []
        self.nonce_updates: Dict[str, int] = {}
        self.orders_closed_at = time.time()
        self.fill_start_time = time.time()
    
    def can_add_order(self, order: SimulatedOrder) -> bool:
        """Check if order can be added without exceeding gas limits."""
        return (self.gas_used + order.sim_value.gas_used <= self.context.block_gas_limit)
    
    def get_proposer_payout_tx_value(self, fee_recipient: str = None) -> Optional[int]:
        """
        Gets the block profit excluding the expected payout gas that we'll pay.
        
        Args:
            fee_recipient: The fee recipient address (defaults to context coinbase)
            
        Returns:
            The true block value after subtracting payout gas cost, or None if profit too low
        """
        if fee_recipient is None:
            fee_recipient = self.context.coinbase
        
        payout_gas_cost = PAYOUT_GAS_LIMIT * self.context.block_base_fee
        
        if self.coinbase_profit >= payout_gas_cost:
            return self.coinbase_profit - payout_gas_cost
        else:
            logger.warning(f"Profit too low to cover payout tx gas: {self.coinbase_profit} < {payout_gas_cost}")
            return None

    def commit_order(self, order: SimulatedOrder, 
                    profit_validator: Callable[[SimValue, SimValue], None]) -> Dict[str, Any]:
        """
        Attempt to commit an order to the block using in-block EVM simulation.
        
        Args:
            order: The order to commit
            profit_validator: Function to validate profit didn't drop too much
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Always use in-block simulation - re-execute the order in current EVM state
            logger.debug(f"Re-executing order {order.order.id()} in block context")
            in_block_result = self.simulator.simulate_and_commit_order(order.order)
            
            if not in_block_result.simulation_result.success:
                raise ExecutionError(
                    f"Order failed in-block execution: {in_block_result.simulation_result.error_message}",
                    "in_block_execution_failed"
                )
            
            # Use in-block simulation results
            simulated_profit = in_block_result.simulation_result.coinbase_profit
            simulated_gas = in_block_result.simulation_result.gas_used
            simulated_blob_gas = in_block_result.simulation_result.blob_gas_used
            simulated_kickbacks = in_block_result.simulation_result.paid_kickbacks
            
            logger.debug(f"In-block execution results for {order.order.id()}: "
                       f"gas={simulated_gas}, profit={simulated_profit}")
            
            new_sim_value = SimValue(
                coinbase_profit=simulated_profit,
                gas_used=simulated_gas,
                blob_gas_used=simulated_blob_gas,
                paid_kickbacks=simulated_kickbacks
            )
            
            # Validate profit hasn't degraded too much
            profit_validator(order.sim_value, new_sim_value)
            
            # Check gas limits
            if self.gas_used + simulated_gas > self.context.block_gas_limit:
                raise ExecutionError(
                    f"Gas limit exceeded: {self.gas_used + simulated_gas} > {self.context.block_gas_limit}",
                    "gas_limit_exceeded"
                )
            
            # Update block state
            self.gas_used += simulated_gas
            self.coinbase_profit += simulated_profit
            self.blob_gas_used += simulated_blob_gas
            self.paid_kickbacks += simulated_kickbacks
            self.included_orders.append(order)
            
            # Update nonces for accounts used by this order
            nonces_updated = []
            for nonce_info in order.order.nonces():
                if not nonce_info.optional:
                    new_nonce = nonce_info.nonce + 1
                    self.nonce_updates[nonce_info.address] = new_nonce
                    nonces_updated.append((nonce_info.address, new_nonce))
            
            logger.debug(
                f"Order {order.order.id()} committed: "
                f"{simulated_gas:,} gas, {simulated_profit} wei profit"
            )
            
            return {
                'success': True,
                'gas_used': simulated_gas,
                'coinbase_profit': simulated_profit,
                'blob_gas_used': simulated_blob_gas,
                'paid_kickbacks': simulated_kickbacks,
                'nonces_updated': nonces_updated,
                'sim_value': new_sim_value
            }
            
        except ExecutionError:
            raise
        except Exception as e:
            logger.debug(f"Order {order.order.id()} execution failed: {e}")
            raise ExecutionError(f"Order execution failed: {e}", "execution_error")
    
    def finalize_block(self) -> BlockResult:
        """
        Finalize the block and return results.
        """
        fill_time_ms = (time.time() - self.fill_start_time) * 1000
        
        # Calculate true block value by subtracting expected payout gas cost
        true_block_value = self.get_proposer_payout_tx_value()
        
        final_bid_value = true_block_value if true_block_value is not None else 0
        
        payout_gas_cost = PAYOUT_GAS_LIMIT * self.context.block_base_fee
        
        logger.info(f"{self.builder_name} block finalized: "
                   f"raw_profit={self.coinbase_profit / 10**18:.6f} ETH, "
                   f"payout_gas_cost={payout_gas_cost / 10**18:.6f} ETH "
                   f"(gas_limit={PAYOUT_GAS_LIMIT}), "
                   f"true_block_value={final_bid_value / 10**18:.6f} ETH")
        
        trace = BlockTrace(
            bid_value=final_bid_value,
            raw_coinbase_profit=self.coinbase_profit,
            payout_gas_cost=payout_gas_cost,
            gas_used=self.gas_used,
            gas_limit=self.context.block_gas_limit,
            blob_gas_used=self.blob_gas_used,
            num_orders=len(self.included_orders),
            orders_closed_at=self.orders_closed_at,
            fill_time_ms=fill_time_ms
        )
        
        return BlockResult(
            builder_name=self.builder_name,
            success=True,
            block_trace=trace,
            included_orders=self.included_orders.copy(),
            build_time_ms=fill_time_ms
        )


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
        logger.info(f"Created {name} with sorting: {config.sorting.value}")
    
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
            
            logger.info(f"Using in-block simulation with titanoboa (reusing simulator)")
            
            # Initialize block building helper with the shared simulator
            helper = BlockBuildingHelper(self.name, evm_simulator)

            evm_simulator.fork_at_block(helper.context.block_number - 1)

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
            
            def profit_validator(original_sim: SimValue, new_sim: SimValue) -> None:
                # Similar to rbuilder's simulation_too_low check
                if self.priority_class.simulation_too_low(original_sim, new_sim):
                    raise ExecutionError(
                        f"Profit too low: {new_sim.coinbase_profit} < expected",
                        "profit_too_low"
                    )
            
            # Attempt to commit the order
            try:
                result = helper.commit_order(sim_order, profit_validator)
                
                # Order succeeded - update nonces in order store
                nonces_updated = []
                for account, nonce in result['nonces_updated']:
                    nonces_updated.append(TxNonce(address=account, nonce=nonce, optional=False))
                
                if nonces_updated:
                    logger.debug(f"Updating nonces after order {order_id}:")
                    for nonce_update in nonces_updated:
                        logger.debug(f"  {nonce_update.address}: -> {nonce_update.nonce}")
                    order_store.update_onchain_nonces(nonces_updated)
                
                logger.debug(
                    f"Included order {order_id}: {result['gas_used']:,} gas, "
                    f"{result['coinbase_profit']} wei profit"
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
            error.error_type == "profit_too_low"):
            
            # Re-insert order with updated simulation value for retry
            # In a real implementation, this would use the actual execution result
            logger.debug(f"Retrying order {order_id} (attempt {attempt_count + 1}/{self.config.failed_order_retries})")
            order_attempts[order_id] += 1
            order_store.insert_order(sim_order)
            
        else:
            # Mark order as permanently failed
            if self.config.drop_failed_orders:
                failed_orders.add(order_id)
            
            logger.debug(
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
