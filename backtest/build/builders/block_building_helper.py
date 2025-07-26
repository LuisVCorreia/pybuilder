from decimal import Decimal
import logging
import copy
import time
from typing import List, Dict, Optional, Callable

from backtest.build.simulation.sim_utils import SimulatedOrder, SimValue
from backtest.build.simulation.evm_simulator import EVMSimulator
from .block_result import BlockResult, BlockTrace
from backtest.common.order import OrderId

logger = logging.getLogger(__name__)

# Assume standard payout tx gas limit
# rbuilder uses more complex logic to estimate this, but for simple backtesting we can use a fixed value
PAYOUT_GAS_LIMIT = 21000  

class ExecutionError(Exception):
    """Exception raised when order execution fails."""
    def __init__(
        self,
        message: str,
        error_type: str = "execution_error",
        new_sim: Optional[SimValue] = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.new_sim    = new_sim

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
        self.simulator = simulator
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
                    filter_result: Optional[Callable] = None) -> tuple[SimValue, List[tuple]]:
        """
        Attempt to commit an order to the block using in-block EVM simulation.
        
        Args:
            order: The order to commit
            filter_result: Optional filter function to validate simulation results
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Always use in-block simulation - re-execute the order in current EVM state
            logger.debug(f"Re-executing order {order.order.id()} in block context")
            in_block_result, checkpoint = self.simulator.simulate_and_commit_order(order.order)
            
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
                mev_gas_price=simulated_profit / simulated_gas if simulated_gas > 0 else Decimal(0),
                paid_kickbacks=simulated_kickbacks
            )

            if filter_result:
                filter_result(order.order.id(), order.sim_value, new_sim_value)

            order.sim_value = new_sim_value
            
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
            
            return new_sim_value, nonces_updated
            
        except ExecutionError:
            self.simulator.vm.state.revert(checkpoint)
            logger.debug(f"Order {order.order.id()} execution failed, rolling back state")
            raise
        except Exception as e:
            self.simulator.vm.state.revert(checkpoint)
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

        orders_deep_copy = [copy.deepcopy(order) for order in self.included_orders]

        return BlockResult(
            builder_name=self.builder_name,
            success=True,
            block_trace=trace,
            included_orders=orders_deep_copy,
            build_time_ms=fill_time_ms
        )
