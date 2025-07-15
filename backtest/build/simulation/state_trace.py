from typing import Dict, List, Set, Any
from dataclasses import dataclass, field

@dataclass
class SlotKey:
    """Key for identifying a storage slot"""
    address: str
    slot: int

    def __hash__(self):
        return hash((self.address, self.slot))

    def __eq__(self, other):
        return isinstance(other, SlotKey) and self.address == other.address and self.slot == other.slot


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


    def merge(self, other: 'UsedStateTrace') -> 'UsedStateTrace':
        """Merge another trace into this one (for bundle orders)"""
        merged = UsedStateTrace()
        
        # Merge storage reads (first read wins)
        merged.read_slot_values = {**self.read_slot_values, **other.read_slot_values}
        # Merge storage writes (last write wins)  
        merged.written_slot_values = {**self.written_slot_values, **other.written_slot_values}
        
        # Merge balance reads (first read wins)
        merged.read_balances = {**self.read_balances, **other.read_balances}
        
        # Accumulate amounts
        merged.received_amount = self._merge_amounts(self.received_amount, other.received_amount)
        merged.sent_amount = self._merge_amounts(self.sent_amount, other.sent_amount)
        
        # Combine contract lists
        merged.created_contracts = list(set(self.created_contracts + other.created_contracts))
        merged.destructed_contracts = list(set(self.destructed_contracts + other.destructed_contracts))
        
        return merged

    def _merge_amounts(self, amounts1: Dict[str, int], amounts2: Dict[str, int]) -> Dict[str, int]:
        """Helper to merge amount dictionaries by summing values"""
        merged = amounts1.copy()
        for addr, amount in amounts2.items():
            merged[addr] = merged.get(addr, 0) + amount
        return merged

    def conflicts_with(self, other: 'UsedStateTrace') -> bool:
        """Check if this trace conflicts with another (for MEV conflict detection)"""
        # Check for overlapping storage writes
        written_keys = set(self.written_slot_values.keys())
        other_written_keys = set(other.written_slot_values.keys())
        if written_keys & other_written_keys:
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
                # Format integers as hex for direct comparison with Rust's B256/U256 output
                output_lines.append(f"  - Address: {slot_key.address}, Slot: {hex(slot_key.slot)}, Value: {hex(value)}")

        if self.written_slot_values:
            output_lines.append("\n[Written Slots]")
            for slot_key, value in self.written_slot_values.items():
                output_lines.append(f"  - Address: {slot_key.address}, Slot: {hex(slot_key.slot)}, Value: {hex(value)}")

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
