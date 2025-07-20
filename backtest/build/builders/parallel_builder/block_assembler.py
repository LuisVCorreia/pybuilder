import logging
from typing import List, Tuple

from backtest.build.simulation.sim_utils import SimulatedOrder
from backtest.build.simulation.evm_simulator import EVMSimulator
from backtest.build.builders.block_result import BlockResult
from .task import ResolutionResult, ConflictGroup
from backtest.build.builders.block_building_helper import BlockBuildingHelper

logger = logging.getLogger(__name__)


class BlockAssembler:
    """
    Assembles the final block for the parallel builder:
      - Takes best (ResolutionResult, ConflictGroup) pairs.
      - Reconstructs the winning per-group order sequences.
      - Concatenates them (groups sorted by descending profit).
      - Re-simulates / commits each order sequentially in a single EVM context
        via BlockBuildingHelper.
    """

    def __init__(self, evm_simulator: EVMSimulator, builder_name: str = "parallel"):
        self.simulator = evm_simulator
        self.builder_name = builder_name

    def assemble_block(
        self,
        best_results: List[Tuple[ResolutionResult, ConflictGroup]]
    ) -> BlockResult:
        """
        Build and finalize the block from best group resolutions.

        Args:
            best_results: List of (ResolutionResult, ConflictGroup) pairs.
        """
        if not best_results:
            return BlockResult(
                builder_name=self.builder_name,
                success=False,
                error_message="No results to assemble"
            )

        # Sort groups by descending total profit
        best_results.sort(key=lambda res: res[0].total_profit, reverse=True)

        helper = BlockBuildingHelper(self.builder_name, self.simulator)
        self.simulator.fork_at_block(self.simulator.context.block_number - 1)

        # Reconstruct full final execution ordering
        final_order_sequence: List[SimulatedOrder] = []
        for result, group in best_results:
            group_orders = self._orders_from_result(result, group)
            if not group_orders:
                continue
            final_order_sequence.extend(group_orders)

        # Commit orders sequentially
        for order in final_order_sequence:
            try:
                helper.commit_order(order)
            except Exception as e:
                logger.error(
                    "Unexpected exception committing order %s: %s",
                    order.order.id(), e,
                    exc_info=True
                )

        # Finalize block result using helper
        return helper.finalize_block()

    def _orders_from_result(
        self,
        result: ResolutionResult,
        group: ConflictGroup
    ) -> List[SimulatedOrder]:
        """
        Map (local_index, profit) entries from resolution result
        back to the group's SimulatedOrder objects in that local order.
        Only include those that were actually attempted/executed in the sequence_of_orders.
        """
        orders: List[SimulatedOrder] = []
        for local_idx, _ in result.sequence_of_orders:
            if 0 <= local_idx < len(group.orders):
                orders.append(group.orders[local_idx])
            else:
                logger.warning(
                    "Resolution result references invalid order index %d (group size %d)",
                    local_idx, len(group.orders)
                )
        return orders
