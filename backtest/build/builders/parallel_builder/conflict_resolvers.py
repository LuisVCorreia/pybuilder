import itertools
import random
import threading
import logging
from typing import List, Dict, Tuple, Optional

from boa.vm.py_evm import Address
from eth.db.diff import DBDiff
from eth.db.account import AccountDB

from backtest.build.simulation.evm_simulator import EVMSimulator
from backtest.build.simulation.sim_utils import SimulatedOrder
from .task import ConflictTask, ResolutionResult, Algorithm
from backtest.common.order import BundleOrder

logger = logging.getLogger(__name__)

class SimulationCacheEntry:
    def __init__(
        self,
        result: ResolutionResult,
        accounts_diff: DBDiff,
        trie_diff: DBDiff,
        storage_diffs: Dict[bytes, Dict[int, int]],
    ):
        self.result = result
        self.accounts_diff = accounts_diff
        self.trie_diff = trie_diff
        self.storage_diffs = storage_diffs


class ConflictResolver:
    """
    Resolves conflicts by trying different orderings and caching partial simulations.
    Cache key = concat of executed order IDs.
    """
    _cache_lock = threading.RLock()
    _simulation_cache: Dict[str, SimulationCacheEntry] = {}

    def __init__(self, evm_simulator: EVMSimulator):
        self.evm_simulator = evm_simulator

    def resolve_conflict_task(self, task: ConflictTask) -> ResolutionResult:
        sequences = self._generate_sequences_to_try(task)
        best = ResolutionResult(total_profit=0, sequence_of_orders=[])

        for seq in sequences:
            key = self._get_cache_key(task.group.orders, seq)
            with ConflictResolver._cache_lock:
                entry = ConflictResolver._simulation_cache.get(key)

            if entry is None:
                entry = self._simulate_sequence_with_cache(task.group.orders, seq)
                with ConflictResolver._cache_lock:
                    ConflictResolver._simulation_cache[key] = entry

            if entry.result.total_profit > best.total_profit:
                best = entry.result

        return best

    def _simulate_sequence_with_cache(
        self,
        orders: List[SimulatedOrder],
        sequence: List[int]
    ) -> SimulationCacheEntry:

        logger.debug("Simulating sequence: %s", sequence)

        # Reset VM to a clean state
        self.evm_simulator.fork_at_block(self.evm_simulator.context.block_number - 1)
        acct_db: AccountDB = self.evm_simulator.env.evm.vm.state._account_db

        # Collect referenced (non-optional) addresses and build initial nonce_state
        referenced = self._collect_referenced_addresses(orders, sequence)
        nonce_state = self._init_nonce_state(referenced)

        # Locate longest cached prefix
        cached_len, cached_entry = self._find_cached_prefix(orders, sequence)

        # Apply cached state if present
        if cached_entry:
            self._apply_cached_entry(acct_db, cached_entry)
            logger.debug(
                "Using prefix %s with profit_seq %s",
                tuple(sequence[:cached_len]),
                cached_entry.result.sequence_of_orders
            )

            remaining = list(sequence[cached_len:])[::-1]
            total_profit = cached_entry.result.total_profit
            seq_profits = cached_entry.result.sequence_of_orders.copy()
            executed_ids: List[int] = list(sequence[:cached_len])
        else:
            remaining = list(sequence)[::-1]
            total_profit = 0
            seq_profits: List[Tuple[int, int]] = []
            executed_ids: List[int] = []

        # Sync nonces from VM after possibly applying cache
        self._sync_nonces_from_vm(referenced, nonce_state)

        pending: List[int] = []

        # Main simulation loop
        while remaining:
            idx = remaining.pop()
            order = orders[idx]

            if not self._are_nonces_valid(order, nonce_state):
                logger.debug("Order %s nonces not valid, deferring", order.order.id().value)
                pending.append(idx)
                continue

            try:
                sim_ord, _ = self.evm_simulator.simulate_and_commit_order(order.order)
            except Exception as exc:
                logger.warning("Simulator exception on order %d: %s", idx, exc)
                seq_profits.append((idx, 0))
                executed_ids.append(idx)
            else:
                profit = sim_ord.sim_value.coinbase_profit
                seq_profits.append((idx, profit))
                executed_ids.append(idx)

                total_profit += profit

                # Reload nonces after state changes
                self._sync_nonces_from_vm(referenced, nonce_state)

                # Try to unlock pending orders
                if pending:
                    still = []
                    for p in pending:
                        if self._are_nonces_valid(orders[p], nonce_state):
                            remaining.append(p)
                        else:
                            still.append(p)
                    pending = still

            # 7) Snapshot/cache current prefix
            self._cache_current_prefix(
                acct_db=acct_db,
                orders=orders,
                executed_ids=executed_ids,
                seq_profits=seq_profits,
                total_profit=total_profit
            )

        if pending:
            # Pending orders can remain if multiple orders have the same nonce
            logger.debug("Leftover pending after sim: %s", pending)

        # Return the last cached entry (the full run)
        prefix_key = self._get_cache_key(orders, executed_ids)
        with ConflictResolver._cache_lock:
            return ConflictResolver._simulation_cache[prefix_key]


    def _find_cached_prefix(
        self, orders: List[SimulatedOrder], sequence: List[int]
    ) -> Tuple[int, Optional[SimulationCacheEntry]]:
        cached_len = 0
        cached_entry = None
        with ConflictResolver._cache_lock:
            for L in range(len(sequence), 0, -1):
                k = self._get_cache_key(orders, sequence[:L])
                e = ConflictResolver._simulation_cache.get(k)
                if e:
                    cached_len, cached_entry = L, e
                    break
        return cached_len, cached_entry

    def _apply_cached_entry(self, acct_db: AccountDB, entry: SimulationCacheEntry) -> None:
        entry.accounts_diff.apply_to(acct_db._journaldb, apply_deletes=True)
        entry.trie_diff.apply_to(acct_db._journaltrie, apply_deletes=True)
        for addr, slot_map in entry.storage_diffs.items():
            store = acct_db._get_address_store(addr)
            for slot, val in slot_map.items():
                store.set(slot, val)
        acct_db._account_cache.clear()

    def _collect_referenced_addresses(
        self, orders: List[SimulatedOrder], sequence: List[int]
    ) -> set:
        referenced = set()
        for idx in sequence:
            for ni in orders[idx].order.nonces():
                if not getattr(ni, "optional", False):
                    referenced.add(ni.address)
        return referenced

    def _init_nonce_state(self, referenced: set) -> Dict[any, int]:
        state: Dict[any, int] = {}
        for raw_addr in referenced:
            can_addr = Address(raw_addr).canonical_address
            state[raw_addr] = self.evm_simulator.env.evm.vm.state.get_nonce(can_addr)
        return state

    def _sync_nonces_from_vm(self, referenced: set, nonce_state: Dict[any, int]) -> None:
        for raw_addr in referenced:
            can = Address(raw_addr).canonical_address
            nonce_state[raw_addr] = self.evm_simulator.env.evm.vm.state.get_nonce(can)

    def _cache_current_prefix(
        self,
        acct_db: AccountDB,
        orders: List[SimulatedOrder],
        executed_ids: List[int],
        seq_profits: List[Tuple[int, int]],
        total_profit: int
    ) -> None:
        prefix_key = self._get_cache_key(orders, executed_ids)

        accounts_diff = acct_db._journaldb.diff()
        trie_diff = acct_db._journaltrie.diff()
        storage_diffs: Dict[bytes, Dict[int, int]] = {}
        for addr, store in acct_db._account_stores.items():
            slots = store.get_accessed_slots()
            slot_map = {s: store.get(s) for s in slots}
            if slot_map:
                storage_diffs[addr] = slot_map

        logger.debug(
            "Caching executed orders %s with profit_seq %s",
            executed_ids,
            seq_profits
        )
        prefix_result = ResolutionResult(
            total_profit=total_profit,
            sequence_of_orders=seq_profits.copy()
        )
        entry = SimulationCacheEntry(prefix_result, accounts_diff, trie_diff, storage_diffs)

        with ConflictResolver._cache_lock:
            ConflictResolver._simulation_cache[prefix_key] = entry

    def _are_nonces_valid(self, order: SimulatedOrder, nonce_state: Dict[any, int]) -> bool:
        for nonce_info in order.order.nonces():
            if getattr(nonce_info, "optional", False):
                continue
            expected = nonce_state.get(nonce_info.address, nonce_info.nonce)
            if nonce_info.nonce != expected:
                logger.debug(f"Order {order.order.id()} has invalid nonce {nonce_info.nonce}, expected: {expected}")
                return False
        return True

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
        return [list(p) for p in itertools.permutations(range(num_orders))]

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

    def _get_order_length(self, order: SimulatedOrder) -> int:
        """Get the length (number of transactions) in an order."""
        # For bundle orders, this would be the number of transactions
        # For simple orders, it's 1
        if isinstance(order.order, BundleOrder):
            return len(order.order.child_orders)
        return 1

    def _get_cache_key(self, orders: List[SimulatedOrder], sequence: List[int]) -> str:
        """Generate cache key for a sequence."""
        order_ids = [str(orders[i].order.id().value) for i in sequence]
        return "|".join(order_ids)
