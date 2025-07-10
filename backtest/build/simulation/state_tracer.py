import logging
from typing import Dict, Set, Optional
from pyrevm import EVM, AccountInfo
from .state_trace import UsedStateTrace, SlotKey

logger = logging.getLogger(__name__)


class StateTracker:
    """
    Tracks state changes during EVM execution to generate UsedStateTrace.
    
    This works by taking snapshots before and after execution and comparing
    the states to derive what was read, written, created, or destroyed.
    """
    
    def __init__(self, evm: EVM):
        self.evm = evm
        self.pre_state: Optional[Dict[str, AccountInfo]] = None
        self.pre_storage: Dict[str, Dict[int, int]] = {}
        self.accessed_storage: Set[SlotKey] = set()
        self.trace = UsedStateTrace()
        
    def start_tracking(self):
        """Start tracking by taking a snapshot of the current state"""
        # Capture initial state
        self.pre_state = dict(self.evm.journal_state)
        self.pre_storage = {}
        self.accessed_storage = set()
        self.trace = UsedStateTrace()
        
        logger.debug(f"Started state tracking with {len(self.pre_state)} accounts")
        
    def track_storage_access(self, address: str, slot: int, value: int, is_write: bool = False):
        """Manually track storage access (call this during simulation if needed)"""
        slot_key = SlotKey(address, slot)
        self.accessed_storage.add(slot_key)
        
        if not is_write and slot_key not in self.trace.read_slot_values:
            # First read of this slot
            self.trace.read_slot_values[slot_key] = value
            
        if is_write:
            # Track write
            self.trace.written_slot_values[slot_key] = value
            
        self.trace.accessed_storage_keys.add(slot_key)
    
    def track_balance_access(self, address: str, balance: int):
        """Track balance access (call this during simulation if needed)"""
        if address not in self.trace.read_balances:
            self.trace.read_balances[address] = balance
        self.trace.accessed_addresses.add(address)
        
    def finish_tracking(self) -> UsedStateTrace:
        """
        Finish tracking by comparing final state with initial state.
        Returns the completed UsedStateTrace.
        """
        if self.pre_state is None:
            logger.warning("finish_tracking called without start_tracking")
            return self.trace
            
        try:
            current_state = dict(self.evm.journal_state)
            self._compare_states(self.pre_state, current_state)
            self._detect_storage_changes()
            
            logger.debug(f"Finished state tracking: {self.trace.summary()}")
            return self.trace
            
        except Exception as e:
            logger.error(f"Error during state tracking: {e}")
            return self.trace
    
    def _compare_states(self, pre_state: Dict[str, AccountInfo], post_state: Dict[str, AccountInfo]):
        """Compare pre and post states to detect changes"""
        
        # Find all addresses that appear in either state
        all_addresses = set(pre_state.keys()) | set(post_state.keys())
        
        for address in all_addresses:
            pre_account = pre_state.get(address)
            post_account = post_state.get(address)
            
            # Track address access
            self.trace.accessed_addresses.add(address)
            
            if pre_account is None and post_account is not None:
                # Account was created
                self.trace.created_contracts.append(address)
                logger.debug(f"Detected account creation: {address}")
                
            elif pre_account is not None and post_account is None:
                # Account was destroyed
                self.trace.destructed_contracts.append(address)
                logger.debug(f"Detected account destruction: {address}")
                
            elif pre_account is not None and post_account is not None:
                # Account existed before and after, check for changes
                self._compare_account_changes(address, pre_account, post_account)
    
    def _compare_account_changes(self, address: str, pre_account: AccountInfo, post_account: AccountInfo):
        """Compare individual account changes"""
        
        # Track balance read (use pre-execution balance as the "read" value)
        if address not in self.trace.read_balances:
            self.trace.read_balances[address] = pre_account.balance
            
        # Track balance changes
        if pre_account.balance != post_account.balance:
            balance_diff = post_account.balance - pre_account.balance
            
            if balance_diff > 0:
                # Account received funds
                self.trace.received_amount[address] = self.trace.received_amount.get(address, 0) + balance_diff
                logger.debug(f"Balance increase for {address}: {balance_diff}")
            else:
                # Account sent funds  
                self.trace.sent_amount[address] = self.trace.sent_amount.get(address, 0) + abs(balance_diff)
                logger.debug(f"Balance decrease for {address}: {abs(balance_diff)}")
        
        # Track code changes (indicates contract creation/modification)
        if pre_account.code != post_account.code:
            if pre_account.code in (None, b"", b"\0") and post_account.code not in (None, b"", b"\0"):
                # Code was added (contract creation)
                if address not in self.trace.created_contracts:
                    self.trace.created_contracts.append(address)
                    logger.debug(f"Detected contract code deployment: {address}")
    
    def _detect_storage_changes(self):
        """
        Detect storage changes by sampling known storage slots.
        Note: This is limited compared to a full trace, but works with current pyrevm capabilities.
        """
        
        # For accounts we've seen, try to detect storage changes by sampling some slots
        for address in self.trace.accessed_addresses:
            self._sample_storage_changes(address)
    
    def _sample_storage_changes(self, address: str, max_slots: int = 10):
        """
        Sample storage slots to detect changes.
        This is a heuristic approach since we don't have full storage tracing.
        """
        try:
            # Sample some common storage slots (0-9)
            for slot in range(min(max_slots, 10)):
                try:
                    current_value = self.evm.storage(address, slot)
                    pre_value = self.pre_storage.get(address, {}).get(slot, 0)
                    
                    slot_key = SlotKey(address, slot)
                    
                    # If we haven't tracked this read yet, record it
                    if slot_key not in self.trace.read_slot_values:
                        self.trace.read_slot_values[slot_key] = pre_value
                        
                    # If value changed, record the write
                    if current_value != pre_value:
                        self.trace.written_slot_values[slot_key] = current_value
                        logger.debug(f"Storage change detected {address}[{slot}]: {pre_value} -> {current_value}")
                        
                    self.trace.accessed_storage_keys.add(slot_key)
                    
                except Exception as e:
                    logger.debug(f"Could not access storage {address}[{slot}]: {e}")
                    continue
                    
        except Exception as e:
            logger.debug(f"Error sampling storage for {address}: {e}")


class TracingEVMWrapper:
    """
    Wrapper around pyrevm.EVM that automatically tracks state changes.
    Drop-in replacement for EVM with automatic tracing.
    """
    
    def __init__(self, evm: EVM, auto_trace: bool = True):
        self.evm = evm
        self.auto_trace = auto_trace
        self.tracker: Optional[StateTracker] = None
        
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped EVM"""
        return getattr(self.evm, name)
    
    def start_tracing(self) -> StateTracker:
        """Start state tracking and return the tracker for manual control"""
        self.tracker = StateTracker(self.evm)
        self.tracker.start_tracking()
        return self.tracker
    
    def message_call(self, **kwargs) -> bytes:
        """Wrapped message_call with automatic tracing"""
        
        # Start tracing if auto_trace is enabled
        if self.auto_trace and self.tracker is None:
            self.start_tracing()
            
        # Execute the call
        result = self.evm.message_call(**kwargs)
        
        # Auto-finish tracing if this was the top-level call
        if self.auto_trace and self.tracker is not None:
            self.tracker.finish_tracking()
            
        return result
    
    def deploy(self, **kwargs) -> str:
        """Wrapped deploy with automatic tracing"""
        
        # Start tracing if auto_trace is enabled  
        if self.auto_trace and self.tracker is None:
            self.start_tracing()
            
        # Execute the deployment
        result = self.evm.deploy(**kwargs)
        
        return result
    
    def finish_tracing(self) -> Optional[UsedStateTrace]:
        """Finish tracing and return the trace"""
        if self.tracker is None:
            return None
            
        trace = self.tracker.finish_tracking()
        self.tracker = None
        return trace
    
    def get_current_trace(self) -> Optional[UsedStateTrace]:
        """Get the current trace without finishing tracking"""
        if self.tracker is None:
            return None
        return self.tracker.trace.copy() if hasattr(self.tracker.trace, 'copy') else self.tracker.trace


def create_tracing_evm(*args, **kwargs) -> TracingEVMWrapper:
    """
    Factory function to create a tracing-enabled EVM.
    Same arguments as pyrevm.EVM constructor.
    """
    # Force tracing to be enabled
    kwargs['tracing'] = True
    
    from pyrevm import EVM
    evm = EVM(*args, **kwargs)
    return TracingEVMWrapper(evm)
