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
    This implementation mirrors the logic of rbuilder's ConflictFinder.
    """
    
    def __init__(self):
        self.group_counter = 0
        
        # Index mappings for fast conflict detection
        self.group_reads: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.group_writes: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.group_balance_reads: Dict[str, List[int]] = defaultdict(list)
        self.group_balance_writes: Dict[str, List[int]] = defaultdict(list)
        self.group_code_writes: Dict[str, List[int]] = defaultdict(list)
        
        # Storage for groups and processed orders
        self.groups: Dict[int, 'GroupData'] = {}
        self.processed_orders: Set[str] = set()

    def add_orders(self, orders: List[SimulatedOrder]) -> List[ConflictGroup]:
        """Add new orders and return updated conflict groups."""
        new_orders = [order for order in orders if str(order.order.id()) not in self.processed_orders]
        
        if not new_orders:
            return self.get_order_groups()
        
        for order in new_orders:
            self._add_single_order(order)
            self.processed_orders.add(str(order.order.id()))
        
        return self.get_order_groups()

    def _add_single_order(self, order: SimulatedOrder):
        """
        Adds a single order by finding conflicts and managing group merges,
        mirroring the rbuilder implementation.
        """
        if not order.used_state_trace:
            self._create_singleton_group(order)
            return

        # 1. Always create a temporary group for the new order.
        new_order_group = self._create_group_data_from_order(order)
        
        # 2. Find all conflicts against existing group indexes.
        conflicting_group_ids = self._find_all_conflicts(order.used_state_trace)

        # 3. Handle merging based on the number of conflicts found.
        if not conflicting_group_ids:
            # No conflicts: Add the new order's group as a new singleton group.
            group_id = self.group_counter
            self.group_counter += 1
            self.groups[group_id] = new_order_group
            self._add_group_to_index(group_id, new_order_group)

        elif len(conflicting_group_ids) == 1:
            # One conflict: Merge the new order's group with the existing one under the *same old ID*.
            group_id = conflicting_group_ids.pop()
            other_group = self.groups.pop(group_id)
            self._remove_group_from_index(group_id, other_group)
            
            combined_group = self._combine_groups([new_order_group, other_group])
            self.groups[group_id] = combined_group
            self._add_group_to_index(group_id, combined_group)

        else:
            # Multiple conflicts: Merge all conflicting groups and the new order's group under a *new ID*.
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

    def _find_all_conflicts(self, trace: UsedStateTrace) -> Set[int]:
        """Finds all conflicts for a given trace, mirroring rbuilder's comprehensive checks."""
        conflicts = set()
        
        # Storage Read Conflicts (RAW, Read-vs-CodeWrite)
        for slot_key in trace.read_slot_values.keys():
            conflicts.update(self.group_writes[slot_key.address].get(slot_key.slot, []))
            conflicts.update(self.group_code_writes.get(slot_key.address, []))

        # Storage Write Conflicts (WAR, WAW, Write-vs-CodeWrite)
        for slot_key in trace.written_slot_values.keys():
            conflicts.update(self.group_reads[slot_key.address].get(slot_key.slot, []))
            conflicts.update(self.group_writes[slot_key.address].get(slot_key.slot, []))
            conflicts.update(self.group_code_writes.get(slot_key.address, []))

        # Balance Read Conflicts (RAW)
        for address in trace.read_balances.keys():
            conflicts.update(self.group_balance_writes.get(address, []))
            
        # Balance Write Conflicts (WAR, WAW)
        balance_write_addresses = set(trace.received_amount.keys()) | set(trace.sent_amount.keys())
        for address in balance_write_addresses:
            conflicts.update(self.group_balance_reads.get(address, []))
            conflicts.update(self.group_balance_writes.get(address, []))

        # Code Write Conflicts (CodeWrite vs all)
        code_write_addresses = set(trace.created_contracts) | set(trace.destructed_contracts)
        for address in code_write_addresses:
            conflicts.update(self.group_code_writes.get(address, []))
            for group_ids in self.group_reads.get(address, {}).values():
                conflicts.update(group_ids)
            for group_ids in self.group_writes.get(address, {}).values():
                conflicts.update(group_ids)
        
        return conflicts

    def _create_singleton_group(self, order: SimulatedOrder):
        """Helper to create a new group for an order with no conflicts."""
        group_id = self.group_counter
        self.group_counter += 1
        group_data = self._create_group_data_from_order(order)
        self.groups[group_id] = group_data
        self._add_group_to_index(group_id, group_data)

    def _combine_groups(self, groups: List['GroupData'], removed_group_ids: Set[int] = None) -> 'GroupData':
        """Combines multiple GroupData objects into one, deduplicating traces."""
        combined = GroupData(conflicting_group_ids=removed_group_ids or set())
        
        all_orders, all_reads, all_writes, all_balance_reads, all_balance_writes, all_code_writes = [], [], [], [], [], []
        
        for group in groups:
            all_orders.extend(group.orders)
            all_reads.extend(group.reads)
            all_writes.extend(group.writes)
            all_balance_reads.extend(group.balance_reads)
            all_balance_writes.extend(group.balance_writes)
            all_code_writes.extend(group.code_writes)
            combined.conflicting_group_ids.update(group.conflicting_group_ids)

        combined.orders = all_orders
        combined.reads = list(set(all_reads))
        combined.writes = list(set(all_writes))
        combined.balance_reads = list(set(all_balance_reads))
        combined.balance_writes = list(set(all_balance_writes))
        combined.code_writes = list(set(all_code_writes))

        return combined

    def _create_group_data_from_order(self, order: SimulatedOrder) -> 'GroupData':
        """Creates a GroupData object from a single simulated order."""
        trace = order.used_state_trace
        return GroupData(
            orders=[order],
            reads=list(trace.read_slot_values.keys()) if trace else [],
            writes=list(trace.written_slot_values.keys()) if trace else [],
            balance_reads=list(trace.read_balances.keys()) if trace else [],
            balance_writes=self._get_balance_writes(trace) if trace else [],
            code_writes=trace.created_contracts + trace.destructed_contracts if trace else [],
        )

    def _get_balance_writes(self, trace: UsedStateTrace) -> List[str]:
        """Helper to get all addresses with balance modifications."""
        return list(set(trace.received_amount.keys()) | set(trace.sent_amount.keys()))

    def _add_group_to_index(self, group_id: int, group_data: 'GroupData'):
        """Adds a group's trace data to the conflict detection indexes."""
        for slot_key in group_data.reads:
            self.group_reads[slot_key.address][slot_key.slot].append(group_id)
        for slot_key in group_data.writes:
            self.group_writes[slot_key.address][slot_key.slot].append(group_id)
        for address in group_data.balance_reads:
            self.group_balance_reads[address].append(group_id)
        for address in group_data.balance_writes:
            self.group_balance_writes[address].append(group_id)
        for address in group_data.code_writes:
            self.group_code_writes[address].append(group_id)

    def _remove_group_from_index(self, group_id: int, group_data: 'GroupData'):
        """Removes a group's trace data from the indexes, typically before a merge."""
        for slot_key in group_data.reads:
            try: self.group_reads[slot_key.address][slot_key.slot].remove(group_id)
            except (KeyError, ValueError): pass
        for slot_key in group_data.writes:
            try: self.group_writes[slot_key.address][slot_key.slot].remove(group_id)
            except (KeyError, ValueError): pass
        for address in group_data.balance_reads:
            try: self.group_balance_reads[address].remove(group_id)
            except (KeyError, ValueError): pass
        for address in group_data.balance_writes:
            try: self.group_balance_writes[address].remove(group_id)
            except (KeyError, ValueError): pass
        for address in group_data.code_writes:
            try: self.group_code_writes[address].remove(group_id)
            except (KeyError, ValueError): pass

    def get_order_groups(self) -> List[ConflictGroup]:
        """Returns the current state of conflict groups."""
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
    """Internal data structure for managing group information."""
    orders: List[SimulatedOrder] = field(default_factory=list)
    reads: List[SlotKey] = field(default_factory=list)
    writes: List[SlotKey] = field(default_factory=list)
    balance_reads: List[str] = field(default_factory=list)
    balance_writes: List[str] = field(default_factory=list)
    code_writes: List[str] = field(default_factory=list)
    conflicting_group_ids: Set[int] = field(default_factory=set)
