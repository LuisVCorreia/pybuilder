from typing import List, Tuple
import logging

from backtest.build.simulation.sim_utils import SimulatedOrder
from backtest.build.builders.block_result import BlockResult, BlockTrace
from .task import ResolutionResult, ConflictGroup

logger = logging.getLogger(__name__)


class BlockAssembler:
    """Assembles final blocks from best conflict resolution results."""
    
    def __init__(self, builder_name: str = "parallel"):
        self.builder_name = builder_name

    def assemble_block(self, best_results: List[Tuple[ResolutionResult, ConflictGroup]]) -> BlockResult:
        """Assemble a block from the best results of each conflict group."""
        try:
            if not best_results:
                return BlockResult(
                    builder_name=self.builder_name,
                    success=False,
                    error_message="No results to assemble"
                )

            # Collect all orders in optimal sequence from each group
            all_orders = []
            total_profit = 0
            total_gas_used = 0
            
            for result, group in best_results:
                # Add orders from this group in the optimal sequence
                group_orders = self._extract_orders_from_result(result, group)
                all_orders.extend(group_orders)
                total_profit += result.total_profit
                total_gas_used += sum(order.sim_value.gas_used for order in group_orders)

            # Create block trace
            block_trace = BlockTrace(
                bid_value=total_profit,  # True bid value
                gas_used=total_gas_used,
                gas_limit=30_000_000,  # Standard block gas limit
                blob_gas_used=0,
                num_orders=len(all_orders),
                orders_closed_at=0.0,  # Will be set by caller
                fill_time_ms=0.0,      # Will be set by caller
                raw_coinbase_profit=total_profit,
                payout_gas_cost=0      # Simplified for now
            )

            logger.info(
                f"Assembled block with {len(all_orders)} orders, "
                f"profit: {total_profit / 1e18:.6f} ETH, "
                f"gas: {total_gas_used:,}"
            )

            return BlockResult(
                builder_name=self.builder_name,
                success=True,
                block_trace=block_trace,
                included_orders=all_orders
            )

        except Exception as e:
            logger.error(f"Error assembling block: {e}")
            return BlockResult(
                builder_name=self.builder_name,
                success=False,
                error_message=str(e)
            )

    def _extract_orders_from_result(self, result: ResolutionResult, group: ConflictGroup) -> List[SimulatedOrder]:
        """Extract orders from a resolution result in the optimal sequence."""
        orders = []
        
        # Sort by the sequence specified in the result
        for order_idx, _ in result.sequence_of_orders:
            if order_idx < len(group.orders):
                orders.append(group.orders[order_idx])
            else:
                logger.warning(f"Invalid order index {order_idx} in group of size {len(group.orders)}")
        
        return orders

    def _validate_results(self, best_results: List[Tuple[ResolutionResult, ConflictGroup]]) -> bool:
        """Validate that results are consistent and don't conflict."""
        # Check for conflicts between groups (shouldn't happen if grouping worked correctly)
        all_used_addresses = set()
        
        for result, group in best_results:
            for order in group.orders:
                if order.used_state_trace:
                    # Check for storage conflicts
                    written_slots = set(order.used_state_trace.written_slot_values.keys())
                    if written_slots & all_used_addresses:
                        logger.warning("Detected potential conflict between groups")
                        return False
                    all_used_addresses.update(written_slots)
        
        return True
