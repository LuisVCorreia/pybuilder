from typing import Dict, List, Set, Any
from dataclasses import dataclass, field
import copy

@dataclass
class SlotKey:
    """Key for identifying a storage slot"""
    address: str
    key: int

    def __hash__(self):
        return hash((self.address, self.key))

    def __eq__(self, other):
        return isinstance(other, SlotKey) and self.address == other.address and self.key == other.key


@dataclass
class UsedStateTrace:
    """
    Python equivalent of rbuilder's UsedStateTrace.
    Tracks all state changes during transaction execution.
    
    Limitations (same as rbuilder):
    * written_slot_values, received_amount and sent_amount are not correct if transaction reverts
    """
    # Storage tracking
    read_slot_values: Dict[SlotKey, int] = field(default_factory=dict)
    written_slot_values: Dict[SlotKey, int] = field(default_factory=dict)
    
    # Balance tracking  
    read_balances: Dict[str, int] = field(default_factory=dict)
    received_amount: Dict[str, int] = field(default_factory=dict)  # wei received during execution
    sent_amount: Dict[str, int] = field(default_factory=dict)      # wei sent during execution
    
    # Contract lifecycle
    created_contracts: List[str] = field(default_factory=list)
    destructed_contracts: List[str] = field(default_factory=list)

    def copy(self) -> 'UsedStateTrace':
        """Creates a deep copy of this trace object."""
        return UsedStateTrace(
            read_slot_values=copy.copy(self.read_slot_values),
            written_slot_values=copy.copy(self.written_slot_values),
            read_balances=copy.copy(self.read_balances),
            received_amount=copy.copy(self.received_amount),
            sent_amount=copy.copy(self.sent_amount),
            created_contracts=copy.copy(self.created_contracts),
            destructed_contracts=copy.copy(self.destructed_contracts),
        )

    def append_trace(self, other: 'UsedStateTrace'):
        """
        Appends another trace into this one in-place.
        Order matters: `other` trace is assumed to come after `self`.
        - First read wins for storage and balances.
        - Last write wins for storage.
        - Amounts are accumulated.
        - Contract lists are extended without duplicates.
        """
        # --- Storage Reads (First read wins) ---
        for slot_key, value in other.read_slot_values.items():
            if slot_key not in self.read_slot_values:
                self.read_slot_values[slot_key] = value

        # --- Storage Writes (Last write wins) ---
        self.written_slot_values.update(other.written_slot_values)

        # --- Balance Reads (First read wins) ---
        for address, balance in other.read_balances.items():
            if address not in self.read_balances:
                self.read_balances[address] = balance

        # --- Amount Accumulation ---
        for address, amount in other.received_amount.items():
            self.received_amount[address] = self.received_amount.get(address, 0) + amount
        
        for address, amount in other.sent_amount.items():
            self.sent_amount[address] = self.sent_amount.get(address, 0) + amount

        # --- Contract Lists (Extend without duplicates) ---
        for address in other.created_contracts:
            if address not in self.created_contracts:
                self.created_contracts.append(address)
        
        for address in other.destructed_contracts:
            if address not in self.destructed_contracts:
                self.destructed_contracts.append(address)

    def conflicts_with(self, other: 'UsedStateTrace') -> bool:
        """Check if this trace conflicts with another (for MEV conflict detection)"""
        # Check for overlapping storage writes
        if set(self.written_slot_values.keys()) & set(other.written_slot_values.keys()):
            return True
            
        # Check for overlapping balance changes
        balance_changed = set(self.received_amount.keys()) | set(self.sent_amount.keys())
        other_balance_changed = set(other.received_amount.keys()) | set(other.sent_amount.keys())
        if balance_changed & other_balance_changed:
            return True
            
        # Check for overlapping contract creation/destruction
        if (set(self.created_contracts) & set(other.created_contracts) or
            set(self.destructed_contracts) & set(other.destructed_contracts)):
            return True
            
        return False

    def summary(self) -> str:
        """
        Generates a readable, multi-line string summary of the state trace.
        """
        output_lines = ["\n--- Dumping UsedStateTrace (Python) ---"]

        if self.read_slot_values:
            output_lines.append("\n[Read Slots]")
            for slot_key, value in self.read_slot_values.items():
                output_lines.append(f"  - Address: {slot_key.address}, Slot: {hex(slot_key.key)}, Value: {hex(value)}")

        if self.written_slot_values:
            output_lines.append("\n[Written Slots]")
            for slot_key, value in self.written_slot_values.items():
                output_lines.append(f"  - Address: {slot_key.address}, Slot: {hex(slot_key.key)}, Value: {hex(value)}")

        if self.read_balances:
            output_lines.append("\n[Read Balances]")
            for address, balance in self.read_balances.items():
                output_lines.append(f"  - Address: {address}, Balance: {balance}")

        if self.received_amount:
            output_lines.append("\n[Received Amounts (Wei)]")
            for address, amount in self.received_amount.items():
                output_lines.append(f"  - Address: {address}, Amount: {amount}")

        if self.sent_amount:
            output_lines.append("\n[Sent Amounts (Wei)]")
            for address, amount in self.sent_amount.items():
                output_lines.append(f"  - Address: {address}, Amount: {amount}")

        if self.created_contracts:
            output_lines.append("\n[Created Contracts]")
            for address in self.created_contracts:
                output_lines.append(f"  - Address: {address}")

        if self.destructed_contracts:
            output_lines.append("\n[Destructed Contracts]")
            for address in self.destructed_contracts:
                output_lines.append(f"  - Address: {address}")

        output_lines.append("--- End of Trace Dump ---\n")
        return "\n".join(output_lines)
