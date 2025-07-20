import itertools
import random
from typing import List, Dict
import logging

from boa.vm.py_evm import Address

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
        """
        Simulate a sequence of orders honoring nonce dependencies.

        Rules:
        - Defer orders whose required nonces are not yet met (nonce gap) -> pending_orders.
        - Execute ready orders immediately.
        - If simulate_and_commit_order returns success=True, ALWAYS advance the nonces of its
            non-optional tx senders, even if coinbase profit is zero (revert or neutral effect).
        - A simulator exception counts as a failed attempt: record zero profit, do NOT advance nonce.
        - Only orders that were actually executed (attempted) are recorded in sequence_of_orders.
        """
        try:
            # Start from clean base for this permutation
            self.evm_simulator._fork_at_block(self.evm_simulator.context.block_number - 1)

            # Gather initial nonces from chain for all involved (non-optional) addresses
            referenced_addresses = set()
            for idx in sequence:
                for nonce_info in orders[idx].order.nonces():
                    if not getattr(nonce_info, "optional", False):
                        referenced_addresses.add(nonce_info.address)

            from boa.vm.py_evm import Address
            nonce_state: Dict[str, int] = {
                addr: self.evm_simulator.env.evm.vm.state.get_nonce(Address(addr).canonical_address)
                for addr in referenced_addresses
            }

            # Use stack for remaining order indices (reverse the given sequence)
            remaining_orders = list(sequence)[::-1]
            pending_orders: List[int] = []

            sequence_profits: List[tuple[int, int]] = []
            total_profit = 0

            while remaining_orders:
                order_idx = remaining_orders.pop()
                order = orders[order_idx]

                # If nonces not ready, defer
                if not self._are_nonces_valid(order, nonce_state):
                    pending_orders.append(order_idx)
                    continue

                # Execute
                try:
                    simulated_order = self.evm_simulator.simulate_and_commit_order(order.order)
                except Exception as exc:
                    logger.warning(f"Simulator exception executing order {order_idx}: {exc}")
                    # Record attempt with zero profit; no nonce advance
                    sequence_profits.append((order_idx, 0))
                    continue

                # Extract profit (may be zero)
                order_profit = simulated_order.sim_value.coinbase_profit
                sequence_profits.append((order_idx, order_profit))

                if not simulated_order._error_result:
                    # Count profit (only positive adds to total_profit)
                    if order_profit > 0:
                        total_profit += order_profit

                    # Advance nonces for all non-optional nonce slots
                    updated_accounts = set()
                    for nonce_info in order.order.nonces():
                        if getattr(nonce_info, "optional", False):
                            continue
                        expected = nonce_state.get(nonce_info.address, nonce_info.nonce)
                        # expected should normally equal nonce_info.nonce
                        if nonce_info.nonce == expected:
                            nonce_state[nonce_info.address] = nonce_info.nonce + 1
                        else:
                            # Maintain monotonicity if off
                            nonce_state[nonce_info.address] = max(expected, nonce_info.nonce + 1)
                        updated_accounts.add(nonce_info.address)

                    # Re-check pending orders for unlock
                    if pending_orders:
                        still_pending = []
                        for p_idx in pending_orders:
                            p_order = orders[p_idx]
                            if self._are_nonces_valid(p_order, nonce_state):
                                remaining_orders.append(p_idx)
                            else:
                                still_pending.append(p_idx)
                        pending_orders = still_pending
                else:
                    # success == False should not happen per upstream invariant;
                    # if it does, treat like failed validation: no nonce advance.
                    logger.warning(f"Unexpected validation failure for order {order_idx} (not advancing nonce).")

            # By invariant, pending_orders should be empty now
            if pending_orders:
                logger.warning("Unexpected leftover pending orders after simulation: %s", pending_orders)

            return ResolutionResult(
                total_profit=total_profit,
                sequence_of_orders=sequence_profits
            )

        except Exception as e:
            logger.warning(f"Error simulating sequence {sequence}: {e}", exc_info=True)
            return ResolutionResult(total_profit=0, sequence_of_orders=[])


    def _extract_sequence_nonces(self, orders: List[SimulatedOrder], sequence: List[int]) -> List:
        """
        Extract initial nonce state for orders in the sequence.
        Similar to ordering builder's _extract_initial_nonces.
        """
        from backtest.common.order import TxNonce
        
        account_nonces = {}
        
        # Find the minimum nonce for each account across orders in sequence
        for order_idx in sequence:
            order = orders[order_idx]
            for nonce_info in order.order.nonces():
                account = nonce_info.address
                nonce = nonce_info.nonce
                
                if account not in account_nonces or nonce < account_nonces[account]:
                    account_nonces[account] = nonce
        
        initial_nonces = []
        for account, nonce in account_nonces.items():
            initial_nonces.append(TxNonce(address=account, nonce=nonce, optional=False))
        
        return initial_nonces

    def _are_nonces_valid(self, order: SimulatedOrder, nonce_state: Dict[str, int]) -> bool:
        """
        Check if an order's nonces are valid given the current nonce state.
        """
        for nonce_info in order.order.nonces():
            if not nonce_info.optional:
                expected_nonce = nonce_state.get(nonce_info.address, nonce_info.nonce)
                if nonce_info.nonce < expected_nonce:
                    # Nonce too low - order already used
                    return False
                elif nonce_info.nonce > expected_nonce:
                    # Nonce too high - there's a gap, order can't execute yet
                    return False
        return True

    def _update_nonce_state(self, order: SimulatedOrder, nonce_state: Dict[str, int]) -> None:
        """
        Update nonce state after successfully executing an order.
        """
        for nonce_info in order.order.nonces():
            if not nonce_info.optional:
                # Increment the nonce for this account
                nonce_state[nonce_info.address] = nonce_info.nonce + 1

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
