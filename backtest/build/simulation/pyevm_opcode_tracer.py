import logging
from boa.vm.utils import to_int, to_bytes
from .state_trace import SlotKey, UsedStateTrace

logger = logging.getLogger(__name__)

def _addr_to_hex(addr) -> str:
    """Normalize any address-like object to 0x-prefixed lowercase hex."""
    if addr is None:
        return ""
    if isinstance(addr, bytes):
        return "0x" + addr.hex().lower()
    if hasattr(addr, "canonical_address"):
        return "0x" + addr.canonical_address.hex().lower()
    if hasattr(addr, "hex"):
        return "0x" + addr.hex().lower()
    s = str(addr).lower()
    return s if s.startswith("0x") else "0x" + s


def _storage_addr_hex(computation) -> str:
    """
    py-evm uses:
      - msg.storage_address for storage context (set even in CREATE*)
      - msg.to is None during init code
    Always prefer storage_address, then fall back to to.
    """
    addr = getattr(computation.msg, "storage_address", None) or computation.msg.to
    return _addr_to_hex(addr) if addr is not None else ""


class StateTraceCollector:
    """
    Collects state traces from various opcode tracers.
    This acts as a central repository for all state changes during execution.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.trace = UsedStateTrace()

    def get_trace(self) -> UsedStateTrace:
        return self.trace


class SloadTracer:
    """Traces SLOAD operations to capture storage reads"""
    mnemonic = "SLOAD"

    def __init__(self, sload_op, collector: StateTraceCollector):
        self.collector = collector
        self.sload = sload_op

    def __call__(self, computation):
        # slot is on top of stack before execution
        slot = to_int(computation._stack.values[-1])

        self.sload(computation)

        value = to_int(computation._stack.values[-1])
        addr_hex = _storage_addr_hex(computation)
        if addr_hex:
            key = SlotKey(addr_hex, slot)
            self.collector.trace.read_slot_values.setdefault(key, value)


class SstoreTracer:
    """Traces SSTORE operations to capture storage writes"""
    mnemonic = "SSTORE"

    def __init__(self, sstore_op, collector: StateTraceCollector):
        self.collector = collector
        self.sstore = sstore_op

    def __call__(self, computation):
        slot  = to_int(computation._stack.values[-1])
        value = to_int(computation._stack.values[-2])

        addr_hex = _storage_addr_hex(computation)
        if addr_hex:
            key = SlotKey(addr_hex, slot)
            # prune "same value" writes
            read_val = self.collector.trace.read_slot_values.get(key)
            if read_val is not None and read_val == value:
                # execute but don't record
                self.sstore(computation)
                self.collector.trace.written_slot_values.pop(key, None)
                return
            self.collector.trace.written_slot_values[key] = value

        self.sstore(computation)


class BalanceTracer:
    """Traces BALANCE operations to capture balance reads"""
    mnemonic = "BALANCE"

    def __init__(self, balance_op, collector: StateTraceCollector):
        self.collector = collector
        self.balance = balance_op

    def __call__(self, computation):
        # address is on stack before BALANCE
        addr_bytes = to_bytes(computation._stack.values[-1])[-20:]

        self.balance(computation)

        balance_value = to_int(computation._stack.values[-1])
        addr_hex = _addr_to_hex(addr_bytes)
        if addr_hex and addr_hex not in self.collector.trace.read_balances:
            self.collector.trace.read_balances[addr_hex] = balance_value


class SelfBalanceTracer:
    """Traces SELFBALANCE operations to capture current contract's balance reads"""
    mnemonic = "SELFBALANCE"

    def __init__(self, selfbalance_op, collector: StateTraceCollector):
        self.collector = collector
        self.selfbalance = selfbalance_op

    def __call__(self, computation):
        # Target address of this frame (same rule as storage address)
        addr_hex = _storage_addr_hex(computation)

        self.selfbalance(computation)

        balance_value = to_int(computation._stack.values[-1])
        if addr_hex and addr_hex not in self.collector.trace.read_balances:
            self.collector.trace.read_balances[addr_hex] = balance_value


class SelfdestructTracer:
    """Traces SELFDESTRUCT operations"""
    mnemonic = "SELFDESTRUCT"

    def __init__(self, selfdestruct_op, collector: StateTraceCollector):
        self.collector = collector
        self.selfdestruct = selfdestruct_op

    def __call__(self, computation):
        destroyed_addr = _addr_to_hex(computation.msg.storage_address)

        self.selfdestruct(computation)

        if destroyed_addr and destroyed_addr not in self.collector.trace.destructed_contracts:
            self.collector.trace.destructed_contracts.append(destroyed_addr)


class Create2Tracer:
    """Traces CREATE2 operations to capture contract creation"""
    mnemonic = "CREATE2"

    def __init__(self, create2_op, collector: StateTraceCollector):
        self.collector = collector
        self.create2 = create2_op

    def __call__(self, computation):
        self.create2(computation)

        # Result (new address) is on stack top if success
        if not computation.is_error and computation._stack.values:
            created_addr_int = to_int(computation._stack.values[-1])
            if created_addr_int != 0:
                created_addr = _addr_to_hex(to_bytes(created_addr_int)[-20:])
                if created_addr and created_addr not in self.collector.trace.created_contracts:
                    self.collector.trace.created_contracts.append(created_addr)


class CreateTracer:
    """Traces CREATE operations to capture contract creation"""
    mnemonic = "CREATE"

    def __init__(self, create_op, collector: StateTraceCollector):
        self.collector = collector
        self.create = create_op

    def __call__(self, computation):
        self.create(computation)

        if not computation.is_error and computation._stack.values:
            created_addr_int = to_int(computation._stack.values[-1])
            if created_addr_int != 0:
                created_addr = _addr_to_hex(to_bytes(created_addr_int)[-20:])
                if created_addr and created_addr not in self.collector.trace.created_contracts:
                    self.collector.trace.created_contracts.append(created_addr)



def patch_evm_opcodes_for_tracing(computation_class, collector: StateTraceCollector):
    """
    Patch EVM opcodes to enable comprehensive state tracing.
    """
    SLOAD       = 0x54
    SSTORE      = 0x55
    BALANCE     = 0x31
    SELFBALANCE = 0x47
    SELFDESTRUCT= 0xFF
    CREATE      = 0xF0
    CREATE2     = 0xF5

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

    patched = [SLOAD, SSTORE, BALANCE, SELFBALANCE, SELFDESTRUCT, CREATE, CREATE2]
    logger.debug(f"Patched {len([op for op in patched if op in opcodes])} opcodes for state tracing")


def record_transaction_nonce(collector: StateTraceCollector, tx_sender: str, current_nonce: int):
    """
    Record transaction nonce as storage read/write like rbuilder does.
    Treat nonce changes as slot 0 operations for the sender.
    """
    addr_str = tx_sender.lower()
    if not addr_str.startswith("0x"):
        addr_str = "0x" + addr_str

    next_nonce = current_nonce + 1
    key = SlotKey(addr_str, 0)
    collector.trace.read_slot_values[key] = current_nonce
    collector.trace.written_slot_values[key] = next_nonce
