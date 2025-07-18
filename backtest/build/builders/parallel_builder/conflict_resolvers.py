import itertools
import random
from typing import List, Dict
import logging

from backtest.build.simulation.evm_simulator import EVMSimulator
from backtest.build.simulation.sim_utils import SimulatedOrder
from .task import ConflictTask, ResolutionResult, Algorithm
from backtest.common.order import BundleOrder


logger = logging.getLogger(__name__)


class ConflictResolver:
    """Resolves conflicts in order groups by finding optimal orderings."""
    
    def __init__(self, evm_simulator: EVMSimulator):
        self.evm_simulator = evm_simulator
        self.simulation_cache: Dict[str, ResolutionResult] = {}

    def resolve_conflict_task(self, task: ConflictTask) -> ResolutionResult:
        """Resolve a conflict task and return the best ordering found."""
        if len(task.group.orders) == 1:
            # Single order - trivial case
            order = task.group.orders[0]
            return ResolutionResult(
                total_profit=order.sim_value.coinbase_profit,
                sequence_of_orders=[(0, order.sim_value.coinbase_profit)]
            )
        
        # Generate sequences to try based on algorithm
        sequences_to_try = self._generate_sequences_to_try(task)
        
        best_result = ResolutionResult(total_profit=0, sequence_of_orders=[])
        
        for sequence in sequences_to_try:
            # Check cache first
            cache_key = self._get_cache_key(task.group.orders, sequence)
            if cache_key in self.simulation_cache:
                result = self.simulation_cache[cache_key]
            else:
                # Simulate this sequence
                result = self._simulate_sequence(task.group.orders, sequence)
                self.simulation_cache[cache_key] = result
            
            # Update best if this is better
            if result.total_profit > best_result.total_profit:
                best_result = result
        
        return best_result

    def _generate_sequences_to_try(self, task: ConflictTask) -> List[List[int]]:
        """Generate order sequences to try based on the algorithm."""
        orders = task.group.orders
        num_orders = len(orders)
        
        if task.algorithm == Algorithm.GREEDY:
            return self._generate_greedy_sequences(orders, reverse=False)
        elif task.algorithm == Algorithm.REVERSE_GREEDY:
            return self._generate_greedy_sequences(orders, reverse=True)
        elif task.algorithm == Algorithm.LENGTH:
            return self._generate_length_based_sequences(orders)
        elif task.algorithm == Algorithm.ALL_PERMUTATIONS:
            return self._generate_all_permutations(num_orders)
        elif task.algorithm == Algorithm.RANDOM:
            return self._generate_random_permutations(
                num_orders, 
                task.random_config.seed, 
                task.random_config.count
            )
        else:
            logger.error(f"Unknown algorithm {task.algorithm}")
            return []

    def _generate_greedy_sequences(self, orders: List[SimulatedOrder], reverse: bool) -> List[List[int]]:
        """Generate greedy sequences based on profit and MEV gas price."""
        sequences = []
        
        # Sort by coinbase profit
        profit_indices = list(range(len(orders)))
        profit_indices.sort(
            key=lambda i: orders[i].sim_value.coinbase_profit, 
            reverse=not reverse
        )
        sequences.append(profit_indices)
        
        # Sort by MEV gas price (profit per gas)
        mev_gas_price_indices = list(range(len(orders)))
        mev_gas_price_indices.sort(
            key=lambda i: orders[i].sim_value.mev_gas_price,
            reverse=not reverse
        )
        sequences.append(mev_gas_price_indices)
        
        return sequences

    def _generate_length_based_sequences(self, orders: List[SimulatedOrder]) -> List[List[int]]:
        """
        Generate a sequence of order indices sorted by bundle length (descending),
        with profit (descending) as a tie-breaker.
        """
        # Create a list of tuples containing the data needed for sorting:
        # (original_index, order_length, order_profit)
        order_data = [
            (
                idx,
                self._get_order_length(orders[idx]),
                orders[idx].sim_value.coinbase_profit
            )
            for idx in range(len(orders))
        ]

        order_data.sort(key=lambda x: (x[1], x[2]), reverse=True)

        length_based_sequence = [item[0] for item in order_data]

        return [length_based_sequence]

    def _generate_all_permutations(self, num_orders: int) -> List[List[int]]:
        """Generate all possible permutations."""
        return list(itertools.permutations(range(num_orders)))

    def _generate_random_permutations(self, num_orders: int, seed: int, count: int) -> List[List[int]]:
        """Generate random permutations."""
        random.seed(seed)
        sequences = []
        
        base_sequence = list(range(num_orders))
        for _ in range(count):
            sequence = base_sequence.copy()
            random.shuffle(sequence)
            sequences.append(sequence)
        
        return sequences

    def _simulate_sequence(self, orders: List[SimulatedOrder], sequence: List[int]) -> ResolutionResult:
        """Simulate a specific sequence of orders."""
        try:
            # For now, we'll use the individual order profits
            # In a full implementation, we'd re-simulate the sequence to get accurate profits
            total_profit = 0
            sequence_profits = []
            
            for order_idx in sequence:
                order = orders[order_idx]
                profit = order.sim_value.coinbase_profit
                total_profit += profit
                sequence_profits.append((order_idx, profit))
            
            return ResolutionResult(
                total_profit=total_profit,
                sequence_of_orders=sequence_profits
            )
            
        except Exception as e:
            logger.warning(f"Error simulating sequence {sequence}: {e}")
            return ResolutionResult(total_profit=0, sequence_of_orders=[])

    def _get_order_length(self, order: SimulatedOrder) -> int:
        """Get the length (number of transactions) in an order."""
        # For bundle orders, this would be the number of transactions
        # For simple orders, it's 1
        if isinstance(order.order, BundleOrder):
            return len(order.order.child_orders)
        return 1

    def _get_cache_key(self, orders: List[SimulatedOrder], sequence: List[int]) -> str:
        """Generate cache key for a sequence."""
        order_ids = [str(orders[i].order.id()) for i in sequence]
        return "|".join(order_ids)
