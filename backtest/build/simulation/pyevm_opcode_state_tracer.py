import logging
from boa.environment import Env
from eth.abc import SignedTransactionAPI

from .state_trace import UsedStateTrace
from .pyevm_opcode_tracer import StateTraceCollector, patch_evm_opcodes_for_tracing, record_transaction_nonce

logger = logging.getLogger(__name__)

class PyEVMOpcodeStateTracer:
    """
    State tracer that uses opcode-level patching to capture state changes.
    """
    
    def __init__(self, env: Env):
        self.env = env
        self.collector = StateTraceCollector()
        self._original_opcodes = None
        self._patched = False
    
    def start_tracing(self, tx: SignedTransactionAPI) -> None:
        """
        Start tracing by patching EVM opcodes.
        """
        self.collector.reset()
        
        computation_class = self.env.evm.vm.state.computation_class
        
        if self._original_opcodes is None:
            self._original_opcodes = computation_class.opcodes.copy()
        
        if not self._patched:
            patch_evm_opcodes_for_tracing(computation_class, self.collector)
            self._patched = True
            logger.debug("EVM opcodes patched for tracing")

        self._record_tx_nonce(tx)
    
    def cleanup(self) -> None:
        """
        Restore original opcodes to clean up tracing.
        """
        if self._patched and self._original_opcodes:
            computation_class = self.env.evm.vm.state.computation_class
            computation_class.opcodes = self._original_opcodes.copy()
            self._patched = False
            logger.debug("EVM opcodes restored to original state")
    
    def _record_tx_nonce(self, tx: SignedTransactionAPI) -> None:
        """
        Record transaction nonce as storage read/write.
        """
        try:
            sender = tx.sender
            if sender is None:
                return
            
            sender_str = '0x' + sender.hex().lower()

            logger.debug(f"Recording nonce for sender {sender_str}")

            current_nonce = getattr(tx, 'nonce', 0)
            record_transaction_nonce(self.collector, sender_str, current_nonce)
            
        except Exception as e:
            logger.debug(f"Error recording transaction nonce: {e}")
    
    def finish_tracing(self, computation) -> UsedStateTrace:
        """
        Complete tracing and return the collected state trace.
        """
        trace = self.collector.get_trace()
        try:
            if computation:
                self._extract_message_value_transfer(computation, trace)
                self._process_computation_recursively(computation, trace)

            return trace
            
        except Exception as e:
            logger.error(f"Error during opcode-based state tracing: {e}")
            return trace

    def _process_computation_recursively(self, computation, trace: UsedStateTrace) -> None:
        """
        Recursively walk the computation tree to find contract creations.
        Value transfers are now handled by live tracers.
        """
        try:

            # Record created contracts if this computation is a contract creation
            if hasattr(computation, 'msg') and computation.msg.is_create:
                created_addr = self._addr_to_hex(computation.msg.storage_address)
                if created_addr and created_addr not in trace.created_contracts:
                    trace.created_contracts.append(created_addr)

            # Recursively process child computations
            if hasattr(computation, 'children'):
                for child in computation.children:
                    self._process_computation_recursively(child, trace)
    
        except Exception as e:
            logger.debug(f"Error processing computation for additional info: {e}")

    def _extract_message_value_transfer(self, computation, trace: UsedStateTrace) -> None:
        """
        Extract top-level message value transfer from the computation.
        """
        try:
            if hasattr(computation, 'msg') and hasattr(computation.msg, 'value'):
                value = computation.msg.value
                should_transfer_value = computation.msg.should_transfer_value

                if should_transfer_value and value > 0:
                    # Get sender and receiver
                    sender = getattr(computation.msg, 'sender', None)
                    to = getattr(computation.msg, 'to', None)
                    
                    if sender and to:
                        sender_str = self._addr_to_hex(sender)
                        trace.sent_amount[sender_str] = trace.sent_amount.get(sender_str, 0) + value

                        to_str = self._addr_to_hex(to)
                        trace.received_amount[to_str] = trace.received_amount.get(to_str, 0) + value

        except Exception as e:
            logger.debug(f"Error extracting message value transfer: {e}")
    
    def _addr_to_hex(self, addr) -> str:
        """Convert address to standardized hex string"""
        if isinstance(addr, str):
            addr_str = addr.lower()
            if not addr_str.startswith('0x'):
                addr_str = '0x' + addr_str
            return addr_str
        elif isinstance(addr, bytes):
            return '0x' + addr.hex().lower()
        else:
            addr_str = str(addr).lower()
            if not addr_str.startswith('0x'):
                addr_str = '0x' + addr_str
            return addr_str
