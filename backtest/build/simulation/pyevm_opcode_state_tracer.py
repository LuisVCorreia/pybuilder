import logging
from boa.environment import Env
from eth.abc import SignedTransactionAPI

from .state_trace import UsedStateTrace
from .pyevm_opcode_tracer import StateTraceCollector, patch_evm_opcodes_for_tracing, record_transaction_nonce

logger = logging.getLogger(__name__)


class PyEVMOpcodeStateTracer:
    """
    State tracer that uses opcode-level patching to capture all state changes.
    This is much more accurate than manual before/after comparisons.
    """
    
    def __init__(self, env: Env):
        self.env = env
        self.collector = StateTraceCollector()
        self._original_opcodes = None
        self._patched = False
    
    def start_tracing(self, tx: SignedTransactionAPI) -> None:
        """
        Start tracing by patching EVM opcodes to capture state changes.
        """
        self.collector.reset()
        
        computation_class = self.env.evm.vm.state.computation_class
        
        if self._original_opcodes is None:
            self._original_opcodes = computation_class.opcodes.copy()
        
        if not self._patched:
            patch_evm_opcodes_for_tracing(computation_class, self.collector)
            self._patched = True
            logger.debug("EVM opcodes patched for state tracing")
        
        self._record_tx_nonce(tx)

    def _extract_sender(self, tx: SignedTransactionAPI):
        """Extract sender from transaction using various methods"""
        if hasattr(tx, 'sender'):
            return tx.sender
        elif hasattr(tx, 'signer'):
            return tx.signer()
        elif hasattr(tx, 'from_'):
            return tx.from_
        return None
    
    def start_tracing_for_computation_class(self, computation_class) -> None:
        """
        Start tracing by patching a specific computation class.
        """
        self.collector.reset()
        
        if self._original_opcodes is None:
            self._original_opcodes = computation_class.opcodes.copy()
        
        if not self._patched:
            patch_evm_opcodes_for_tracing(computation_class, self.collector)
            self._patched = True
            logger.debug("EVM opcodes patched for state tracing")
    
    def finish_tracing(self, computation, tx: SignedTransactionAPI = None) -> UsedStateTrace:
        """
        Complete tracing and return the collected state trace.
        """
        try:
            trace = self.collector.get_trace()
            
            # Additional processing from computation
            if computation:
                self._process_computation_for_additional_info(computation, trace)
            
            logger.debug(f"Opcode-based state tracing complete: {trace.summary()}")
            return trace
            
        except Exception as e:
            logger.error(f"Error during opcode-based state tracing: {e}")
            return self.collector.get_trace()
    
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
        Record transaction nonce as storage read/write like rbuilder does.
        """
        try:
            sender = tx.sender
            if sender is None:
                return
            
            if isinstance(sender, str):
                sender_str = sender.lower()
            elif hasattr(sender, 'canonical_address'):
                sender_str = '0x' + sender.canonical_address.hex().lower()
            elif hasattr(sender, 'hex'):
                sender_str = '0x' + sender.hex().lower()
            elif isinstance(sender, bytes):
                sender_str = '0x' + sender.hex().lower()
            else:
                sender_str = str(sender).lower()
            
            if not sender_str.startswith('0x'):
                sender_str = '0x' + sender_str
            
            current_nonce = getattr(tx, 'nonce', 0)
            record_transaction_nonce(self.collector, sender_str, current_nonce)
            
        except Exception as e:
            logger.debug(f"Error recording transaction nonce: {e}")
    
    def _process_computation_for_additional_info(self, computation, trace: UsedStateTrace) -> None:
        """
        Extract any additional information from the computation tree that
        the opcode tracers might have missed.
        """
        try:
            # Process value transfers from message info
            self._extract_message_value_transfers(computation, trace)
            
            # Recursively process child computations
            if hasattr(computation, 'children'):
                for child in computation.children:
                    self._process_computation_for_additional_info(child, trace)
                    
        except Exception as e:
            logger.debug(f"Error processing computation for additional info: {e}")
    
    def _extract_message_value_transfers(self, computation, trace: UsedStateTrace) -> None:
        """
        Extract value transfers from computation messages.
        """
        try:
            if hasattr(computation, 'msg') and hasattr(computation.msg, 'value'):
                value = computation.msg.value
                print("computation.msg:", computation.msg)
                if value > 0:
                    # Get sender and receiver
                    sender = getattr(computation.msg, 'sender', None)
                    to = getattr(computation.msg, 'to', None)

                    print(f"Extracting value transfer: {value} wei from {sender} to {to}")
                    print("THe receiver is:", to)
                    
                    if sender:
                        sender_str = self._addr_to_hex(sender)
                        trace.sent_amount[sender_str] = trace.sent_amount.get(sender_str, 0) + value
                        
                    if to:
                        to_str = self._addr_to_hex(to)
                        trace.received_amount[to_str] = trace.received_amount.get(to_str, 0) + value
                        
        except Exception as e:
            logger.debug(f"Error extracting message value transfers: {e}")
    
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

def create_pyevm_opcode_state_tracer(env: Env) -> PyEVMOpcodeStateTracer:
    """Factory function to create a PyEVM opcode-based state tracer"""
    return PyEVMOpcodeStateTracer(env)
