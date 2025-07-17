from typing import Dict, List, Set
from collections import defaultdict
import logging
from dataclasses import dataclass, field

from backtest.build.simulation.sim_utils import SimulatedOrder
from backtest.build.simulation.state_trace import SlotKey, UsedStateTrace

logger = logging.getLogger(__name__)

@dataclass
class ConflictGroup:
    """Group of conflicting orders that need to be resolved together."""
    id: int
    orders: List[SimulatedOrder]
    conflicting_group_ids: set[int]

class ConflictFinder:
    """
    Finds and manages groups of orders that conflict with each other based on state traces.
    """
    
    def __init__(self):
        self.group_counter = 0
        
        # Index mappings for fast conflict detection
        self.group_reads: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.group_writes: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.group_balance_reads: Dict[str, List[int]] = defaultdict(list)
        self.group_balance_writes: Dict[str, List[int]] = defaultdict(list)
        self.group_code_writes: Dict[str, List[int]] = defaultdict(list)
        
        # Storage for groups
        self.groups: Dict[int, 'GroupData'] = {}

    def add_orders(self, orders: List[SimulatedOrder]) -> List[ConflictGroup]:
        """Processes a list of orders to find and create conflict groups."""
        for order in orders:
            self._add_single_order(order)
        
        return self.get_order_groups()

    def _add_single_order(self, order: SimulatedOrder):
        """
        Adds a single order, finds all conflicts, and merges groups.
        """
        used_state = order.used_state_trace
        if not used_state:
            # If an order has no state trace, it cannot conflict. Create a singleton group.
            self._create_singleton_group(order)
            return

        # Find all conflicts against existing group indexes
        conflicting_group_ids = self._find_all_conflicts(used_state)

        # Create a temporary GroupData object for the new order
        new_order_group = self._create_group_data_from_order(order)

        if not conflicting_group_ids:
            # No conflicts: Add the new order's group as a new singleton group
            group_id = self.group_counter
            self.group_counter += 1
            self.groups[group_id] = new_order_group
            self._add_group_to_index(group_id, new_order_group)

        elif len(conflicting_group_ids) == 1:
            # One conflict: Merge the new order's group with the existing one under the same old ID
            group_id = conflicting_group_ids[0]
            other_group = self.groups.pop(group_id)
            self._remove_group_from_index(group_id, other_group)
            
            combined_group = self._combine_groups([new_order_group, other_group])
            self.groups[group_id] = combined_group
            self._add_group_to_index(group_id, combined_group)

        else:
            # Multiple conflicts: Merge all conflicting groups and the new order's group under a new ID
            groups_to_merge = [new_order_group]
            removed_ids = set()
            
            for group_id in conflicting_group_ids:
                if group_id in self.groups:
                    removed_ids.add(group_id)
                    group_data = self.groups.pop(group_id)
                    self._remove_group_from_index(group_id, group_data)
                    groups_to_merge.append(group_data)
            
            new_group_id = self.group_counter
            self.group_counter += 1
            
            merged_data = self._combine_groups(groups_to_merge, removed_ids)
            self.groups[new_group_id] = merged_data
            self._add_group_to_index(new_group_id, merged_data)

    def _find_all_conflicts(self, trace: UsedStateTrace) -> List[int]:
        """
        Finds all conflicting group IDs based on the new order's state trace.
        """
        conflicts = set()
        
        for slot_key in trace.read_slot_values.keys():
            # Check for read-after-write conflicts
            if slot_key.address in self.group_writes:
                conflicts.update(self.group_writes[slot_key.address].get(slot_key.key, []))
            # Check for reads on contracts being created/destroyed
            conflicts.update(self.group_code_writes.get(slot_key.address, []))

        for slot_key in trace.written_slot_values.keys():
            # Check for write-after-read conflicts
            if slot_key.address in self.group_reads:
                conflicts.update(self.group_reads[slot_key.address].get(slot_key.key, []))
            # Check for writes on contracts being created/destroyed
            conflicts.update(self.group_code_writes.get(slot_key.address, []))

        # Balance Write vs. Balance Read
        balance_write_keys = set(trace.received_amount.keys()) | set(trace.sent_amount.keys())
        for address in balance_write_keys:
            conflicts.update(self.group_balance_reads.get(address, []))

        # Balance Read vs. Balance Write
        for address in trace.read_balances.keys():
            conflicts.update(self.group_balance_writes.get(address, []))

        # Contract Creation/Destruction Conflicts (Code vs. Code/Read/Write)
        code_write_addrs = set(trace.created_contracts) | set(trace.destructed_contracts)
        for address in code_write_addrs:
            # Code-vs-Code
            conflicts.update(self.group_code_writes.get(address, []))
            # Code-vs-Read (any read in the contract)
            if address in self.group_reads:
                for group_ids in self.group_reads[address].values():
                    conflicts.update(group_ids)
            # Code-vs-Write (any write in the contract)
            if address in self.group_writes:
                for group_ids in self.group_writes[address].values():
                    conflicts.update(group_ids)
        
        sorted_conflicts = sorted(list(conflicts))
        return sorted_conflicts

    def _create_singleton_group(self, order: SimulatedOrder):
        """Helper to create a new group for an order."""
        group_id = self.group_counter
        self.group_counter += 1
        group_data = self._create_group_data_from_order(order)
        self.groups[group_id] = group_data
        self._add_group_to_index(group_id, group_data)

    def _combine_groups(self, groups_to_merge: List['GroupData'], removed_group_ids: Set[int] = None) -> 'GroupData':
        """Combines multiple GroupData objects into one, merging their traces and orders."""
        combined = GroupData(conflicting_group_ids=removed_group_ids or set())
        
        all_orders = []
        all_reads = set()
        all_writes = set()
        all_balance_reads = set()
        all_balance_writes = set()
        all_code_writes = set()
        
        for group in groups_to_merge:
            all_orders.extend(group.orders)
            all_reads.update(group.reads)
            all_writes.update(group.writes)
            all_balance_reads.update(group.balance_reads)
            all_balance_writes.update(group.balance_writes)
            all_code_writes.update(group.code_writes)
            combined.conflicting_group_ids.update(group.conflicting_group_ids)

        combined.orders = all_orders
        combined.reads = list(all_reads)
        combined.writes = list(all_writes)
        combined.balance_reads = list(all_balance_reads)
        combined.balance_writes = list(all_balance_writes)
        combined.code_writes = list(all_code_writes)

        return combined

    def _create_group_data_from_order(self, order: SimulatedOrder) -> 'GroupData':
        """Creates a GroupData object from a single simulated order's trace."""
        trace = order.used_state_trace
        if not trace:
            return GroupData(orders=[order])

        balance_writes = sorted(list(set(trace.received_amount.keys()) | set(trace.sent_amount.keys())))
        code_writes = sorted(list(set(trace.created_contracts) | set(trace.destructed_contracts)))

        return GroupData(
            orders=[order],
            reads=list(trace.read_slot_values.keys()),
            writes=list(trace.written_slot_values.keys()),
            balance_reads=list(trace.read_balances.keys()),
            balance_writes=balance_writes,
            code_writes=code_writes,
        )

    def _add_group_to_index(self, group_id: int, group_data: 'GroupData'):
        """Adds a group's trace data to the conflict detection indexes."""
        for slot_key in group_data.reads:
            self.group_reads[slot_key.address][slot_key.key].append(group_id)
        for slot_key in group_data.writes:
            self.group_writes[slot_key.address][slot_key.key].append(group_id)
        for address in group_data.balance_reads:
            self.group_balance_reads[address].append(group_id)
        for address in group_data.balance_writes:
            self.group_balance_writes[address].append(group_id)
        for address in group_data.code_writes:
            self.group_code_writes[address].append(group_id)

    def _remove_group_from_index(self, group_id: int, group_data: 'GroupData'):
        """Removes a group's trace data from the indexes, typically before a merge."""
        for slot_key in group_data.reads:
            if slot_key.address in self.group_reads and slot_key.key in self.group_reads[slot_key.address]:
                try: self.group_reads[slot_key.address][slot_key.key].remove(group_id)
                except ValueError: pass
        for slot_key in group_data.writes:
            if slot_key.address in self.group_writes and slot_key.key in self.group_writes[slot_key.address]:
                try: self.group_writes[slot_key.address][slot_key.key].remove(group_id)
                except ValueError: pass
        for address in group_data.balance_reads:
            if address in self.group_balance_reads:
                try: self.group_balance_reads[address].remove(group_id)
                except ValueError: pass
        for address in group_data.balance_writes:
            if address in self.group_balance_writes:
                try: self.group_balance_writes[address].remove(group_id)
                except ValueError: pass
        for address in group_data.code_writes:
            if address in self.group_code_writes:
                try: self.group_code_writes[address].remove(group_id)
                except ValueError: pass

    def get_order_groups(self) -> List[ConflictGroup]:
        """Returns the current state of conflict groups in the required format."""
        return [
            ConflictGroup(
                id=group_id,
                orders=group_data.orders,
                conflicting_group_ids=group_data.conflicting_group_ids
            )
            for group_id, group_data in self.groups.items()
        ]

@dataclass
class GroupData:
    """Internal data structure for managing group information and merged state traces."""
    orders: List[SimulatedOrder] = field(default_factory=list)
    reads: List[SlotKey] = field(default_factory=list)
    writes: List[SlotKey] = field(default_factory=list)
    balance_reads: List[str] = field(default_factory=list)
    balance_writes: List[str] = field(default_factory=list)
    code_writes: List[str] = field(default_factory=list)
    conflicting_group_ids: Set[int] = field(default_factory=set)
