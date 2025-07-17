import logging
from boa.vm.utils import to_int, to_bytes
from .state_trace import SlotKey, UsedStateTrace

logger = logging.getLogger(__name__)


class StateTraceCollector:
    """
    Collects state traces from various opcode tracers.
    This acts as a central repository for all state changes during execution.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all collected traces"""
        self.trace = UsedStateTrace()
    
    def get_trace(self) -> UsedStateTrace:
        """Get the complete accumulated trace"""
        return self.trace


class SloadTracer:
    """Traces SLOAD operations to capture storage reads"""
    mnemonic = "SLOAD"
    
    def __init__(self, sload_op, collector: StateTraceCollector):
        self.collector = collector
        self.sload = sload_op
    
    def __call__(self, computation):
        # Get slot from stack before calling original SLOAD
        slot = to_int(computation._stack.values[-1])
        
        account = computation.msg.to
        
        self.sload(computation)
        
        value = to_int(computation._stack.values[-1])
        
        if account:
            addr_str = self._addr_to_hex(account)
            slot_key = SlotKey(addr_str, slot)
            
            if slot_key not in self.collector.trace.read_slot_values:
                self.collector.trace.read_slot_values[slot_key] = value
    
    def _addr_to_hex(self, addr) -> str:
        if isinstance(addr, bytes):
            return '0x' + addr.hex().lower()
        elif hasattr(addr, 'canonical_address'):
            return '0x' + addr.canonical_address.hex().lower()
        elif hasattr(addr, 'hex'):
            return '0x' + addr.hex().lower()
        else:
            addr_str = str(addr).lower()
            if not addr_str.startswith('0x'):
                addr_str = '0x' + addr_str
            return addr_str


class SstoreTracer:
    """Traces SSTORE operations to capture storage writes"""
    mnemonic = "SSTORE"
    
    def __init__(self, sstore_op, collector: StateTraceCollector):
        self.collector = collector
        self.sstore = sstore_op
    
    def __call__(self, computation):
        # Get value and slot from stack before calling original SSTORE
        value, slot = [to_int(t) for t in computation._stack.values[-2:]]

        account = computation.msg.to
        
        if account:
            addr_str = self._addr_to_hex(account)
            slot_key = SlotKey(addr_str, slot)

            # Check for same value write before executing SSTORE
            initial_read_value = self.collector.trace.read_slot_values.get(slot_key)
            
            if initial_read_value is not None and initial_read_value == value:
                if slot_key in self.collector.trace.written_slot_values:
                    del self.collector.trace.written_slot_values[slot_key]
                # Execute the SSTORE but don't record it as a state change
                self.sstore(computation)
                return
            
            self.sstore(computation)
            self.collector.trace.written_slot_values[slot_key] = value
    
    def _addr_to_hex(self, addr) -> str:
        if isinstance(addr, bytes):
            return '0x' + addr.hex().lower()
        elif hasattr(addr, 'canonical_address'):
            return '0x' + addr.canonical_address.hex().lower()
        elif hasattr(addr, 'hex'):
            return '0x' + addr.hex().lower()
        else:
            addr_str = str(addr).lower()
            if not addr_str.startswith('0x'):
                addr_str = '0x' + addr_str
            return addr_str


class BalanceTracer:
    """Traces BALANCE operations to capture balance reads"""
    mnemonic = "BALANCE"
    
    def __init__(self, balance_op, collector: StateTraceCollector):
        self.collector = collector
        self.balance = balance_op
    
    def __call__(self, computation):
        # Get address from stack before calling original BALANCE
        addr_bytes = to_bytes(computation._stack.values[-1])[-20:]  # Take last 20 bytes
        
        self.balance(computation)
        
        # Get the balance from stack top (result of BALANCE)
        balance_value = to_int(computation._stack.values[-1])
        
        addr_str = self._addr_to_hex(addr_bytes)
                
        if addr_str not in self.collector.trace.read_balances:
            self.collector.trace.read_balances[addr_str] = balance_value
            
    
    def _addr_to_hex(self, addr) -> str:
        if isinstance(addr, bytes):
            return '0x' + addr.hex().lower()
        return str(addr).lower()


class SelfBalanceTracer:
    """Traces SELFBALANCE operations to capture current contract's balance reads"""
    mnemonic = "SELFBALANCE"
    
    def __init__(self, selfbalance_op, collector: StateTraceCollector):
        self.collector = collector
        self.selfbalance = selfbalance_op
    
    def __call__(self, computation):
        current_address = computation.msg.to
        
        self.selfbalance(computation)
        
        balance_value = to_int(computation._stack.values[-1])
        
        addr_str = self._addr_to_hex(current_address)
                
        if addr_str not in self.collector.trace.read_balances:
            self.collector.trace.read_balances[addr_str] = balance_value
    
    def _addr_to_hex(self, addr) -> str:
        if isinstance(addr, bytes):
            return '0x' + addr.hex().lower()
        return str(addr).lower()


class SelfdestructTracer:
    """Traces SELFDESTRUCT operations"""
    mnemonic = "SELFDESTRUCT"
    
    def __init__(self, selfdestruct_op, collector: StateTraceCollector):
        self.collector = collector
        self.selfdestruct = selfdestruct_op
    
    def __call__(self, computation):
        destroyed_addr = self._addr_to_hex(computation.msg.storage_address)
        
        self.selfdestruct(computation)
        
        if destroyed_addr not in self.collector.trace.destructed_contracts:
            self.collector.trace.destructed_contracts.append(destroyed_addr)

    
    def _addr_to_hex(self, addr) -> str:
        if isinstance(addr, bytes):
            return '0x' + addr.hex().lower()
        return str(addr).lower()


class Create2Tracer:
    """Traces CREATE2 operations to capture contract creation"""
    mnemonic = "CREATE2"
    
    def __init__(self, create2_op, collector: StateTraceCollector):
        self.collector = collector
        self.create2 = create2_op
    
    def __call__(self, computation):
        self.create2(computation)
        
        # Get the created address from stack top (result of CREATE2)
        if not computation.is_error and len(computation._stack.values) > 0:
            created_addr_int = to_int(computation._stack.values[-1])
            if created_addr_int != 0:  # 0 means creation failed
                created_addr = self._addr_to_hex(to_bytes(created_addr_int)[-20:])
                
                if created_addr not in self.collector.trace.created_contracts:
                    self.collector.trace.created_contracts.append(created_addr)
    
    def _addr_to_hex(self, addr) -> str:
        if isinstance(addr, bytes):
            return '0x' + addr.hex().lower()
        return str(addr).lower()


class CreateTracer:
    """Traces CREATE operations to capture contract creation"""
    mnemonic = "CREATE"
    
    def __init__(self, create_op, collector: StateTraceCollector):
        self.collector = collector
        self.create = create_op
    
    def __call__(self, computation):
        self.create(computation)
        
        # Get the created address from stack top (result of CREATE)
        if not computation.is_error and len(computation._stack.values) > 0:
            created_addr_int = to_int(computation._stack.values[-1])
            if created_addr_int != 0:  # 0 means creation failed
                created_addr = self._addr_to_hex(to_bytes(created_addr_int)[-20:])
                
                if created_addr not in self.collector.trace.created_contracts:
                    self.collector.trace.created_contracts.append(created_addr)
    
    def _addr_to_hex(self, addr) -> str:
        if isinstance(addr, bytes):
            return '0x' + addr.hex().lower()
        return str(addr).lower()


def patch_evm_opcodes_for_tracing(computation_class, collector: StateTraceCollector):
    """
    Patch EVM opcodes to enable comprehensive state tracing.
    This follows the same pattern as titanoboa's existing opcode patching.
    """
    # EVM Opcode values
    SLOAD = 0x54
    SSTORE = 0x55
    BALANCE = 0x31
    SELFBALANCE = 0x47
    SELFDESTRUCT = 0xFF
    CREATE = 0xF0
    CREATE2 = 0xF5
    
    # Patch opcodes with our tracers
    opcodes = computation_class.opcodes.copy()
    
    if SLOAD in opcodes:
        opcodes[SLOAD] = SloadTracer(opcodes[SLOAD], collector)
    
    if SSTORE in opcodes:
        opcodes[SSTORE] = SstoreTracer(opcodes[SSTORE], collector)
    
    if BALANCE in opcodes:
        opcodes[BALANCE] = BalanceTracer(opcodes[BALANCE], collector)
    
    if SELFBALANCE in opcodes:
        opcodes[SELFBALANCE] = SelfBalanceTracer(opcodes[SELFBALANCE], collector)
    
    if SELFDESTRUCT in opcodes:
        opcodes[SELFDESTRUCT] = SelfdestructTracer(opcodes[SELFDESTRUCT], collector)
    
    if CREATE in opcodes:
        opcodes[CREATE] = CreateTracer(opcodes[CREATE], collector)
    
    if CREATE2 in opcodes:
        opcodes[CREATE2] = Create2Tracer(opcodes[CREATE2], collector)
    
    computation_class.opcodes = opcodes

    logger.debug(f"Patched {len([op for op in [SLOAD, SSTORE, BALANCE, SELFDESTRUCT, CREATE, CREATE2] if op in opcodes])} opcodes for state tracing")


def record_transaction_nonce(collector: StateTraceCollector, tx_sender: str, current_nonce: int):
    """
    Record transaction nonce as storage read/write like rbuilder does.
    This treats nonce changes as slot 0 operations for the sender.
    """
    addr_str = tx_sender.lower()
    if not addr_str.startswith('0x'):
        addr_str = '0x' + addr_str
    
    next_nonce = current_nonce + 1
    
    # Record as slot 0 read/write
    slot_key = SlotKey(addr_str, 0)
    collector.trace.read_slot_values[slot_key] = current_nonce
    collector.trace.written_slot_values[slot_key] = next_nonce
