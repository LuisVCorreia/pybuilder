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
    
    # Access tracking (for gas optimization)
    accessed_addresses: Set[str] = field(default_factory=set)
    accessed_storage_keys: Set[SlotKey] = field(default_factory=set)

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
        
        # Combine access sets
        merged.accessed_addresses = self.accessed_addresses | other.accessed_addresses
        merged.accessed_storage_keys = self.accessed_storage_keys | other.accessed_storage_keys
        
        return merged

    def _merge_amounts(self, amounts1: Dict[str, int], amounts2: Dict[str, int]) -> Dict[str, int]:
        """Helper to merge amount dictionaries by summing values"""
        merged = amounts1.copy()
        for addr, amount in amounts2.items():
            merged[addr] = merged.get(addr, 0) + amount
        return merged

    def get_total_gas_estimate(self) -> int:
        """Estimate additional gas costs from state access patterns"""
        # This is a rough estimate based on EIP-2929 gas costs
        cold_account_access = len(self.accessed_addresses) * 2600  # COLD_ACCOUNT_ACCESS_COST
        cold_sload = len(self.accessed_storage_keys) * 2100       # COLD_SLOAD_COST
        return cold_account_access + cold_sload

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

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the trace for logging/debugging"""
        return {
            "storage_reads": len(self.read_slot_values),
            "storage_writes": len(self.written_slot_values),
            "balance_reads": len(self.read_balances),
            "accounts_with_received": len(self.received_amount),
            "accounts_with_sent": len(self.sent_amount),
            "contracts_created": len(self.created_contracts),
            "contracts_destructed": len(self.destructed_contracts),
            "total_accessed_addresses": len(self.accessed_addresses),
            "total_accessed_storage": len(self.accessed_storage_keys),
            "estimated_gas_overhead": self.get_total_gas_estimate()
        }
