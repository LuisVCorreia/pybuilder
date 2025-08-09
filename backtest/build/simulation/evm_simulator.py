from decimal import Decimal
import logging
from typing import List
from hexbytes import HexBytes
import ast
import boa_ext

from boa.vm.py_evm import Address
from boa.rpc import EthereumRPC, to_hex
from boa.environment import Env
from eth.abc import SignedTransactionAPI, BlockHeaderAPI
from eth.vm.forks.prague.constants import MAX_BLOB_GAS_PER_BLOCK as PRAGUE_MAX_BLOB_GAS
from eth.vm.forks.cancun.constants import MAX_BLOB_GAS_PER_BLOCK as CANCUN_MAX_BLOB_GAS
from eth_utils.exceptions import ValidationError
from eth.vm.forks.cancun.state import get_total_blob_gas

from backtest.common.order import Order, OrderType, TxOrder, BundleOrder, ShareBundleOrder
from .sim_tree import SimTree, SimulatedResult, NonceKey
from .sim_utils import SimulationContext, SimulatedOrder, SimValue, OrderSimResult, SimulationError
from .pyevm_opcode_state_tracer import PyEVMOpcodeStateTracer
from .state_trace import UsedStateTrace

logger = logging.getLogger(__name__)

class EVMSimulator:
    def __init__(self, simulation_context: SimulationContext, rpc_url: str):
        self.context = simulation_context
        self.rpc = EthereumRPC(rpc_url)
        self.env = Env(fast_mode_enabled=True, fork_try_prefetch_state=True)
        self.rpc_url = rpc_url
        self.vm = self.env.evm.vm
        self.state_tracer = PyEVMOpcodeStateTracer(self.env)
        self.header = self.vm.get_header()
        self.fork_at_block(self.context.block_number - 1)

    def fork_at_block(self, block_number: int):
        # Forking the EVM will re-initiliase the correct VM based on the block timestamp
        block_id = to_hex(block_number)
        # self.env.fork_rpc(self.rpc, block_identifier=block_id, debug=False, cache_dir="./pybuilder_results_3/cache/")
        self.env.fork_rpc(self.rpc, block_identifier=block_id)
        self.vm = self.env.evm.vm
        self.header = self.vm.get_header()
        self._override_execution_context()  # Ensure execution context is set correctly
        self._create_block_header()  


    def simulate_order_with_parents(self, order: Order, parent_orders: List[Order] = None) -> SimulatedOrder:
        parent_orders = parent_orders or []
        try:
            self.fork_at_block(self.context.block_number - 1)
            accumulated_trace = UsedStateTrace()
            # Simulate parent orders and accumulate their traces
            for parent_order in parent_orders:
                self.vm.state.lock_changes()
                parent_res = self._simulate_single_order(parent_order, accumulated_trace)
                if not parent_res.simulation_result.success:
                    error_message = f"Parent order failed: {parent_res.simulation_result.error_message}"
                    error_result = OrderSimResult(success=False, gas_used=0, coinbase_profit=0, blob_gas_used=0, paid_kickbacks=0, error=SimulationError.VALIDATION_ERROR, error_message=error_message)
                    return self._convert_result_to_simulated_order(order, error_result)
                
                # Accumulate parent's state trace
                if parent_res.simulation_result.state_trace:
                    accumulated_trace = parent_res.simulation_result.state_trace

            # Simulate the final order
            self.vm.state.lock_changes()
            final_res = self._simulate_single_order(order, accumulated_trace)
            return final_res
        except Exception as e:
            error_result = OrderSimResult(success=False, gas_used=0, coinbase_profit=0, blob_gas_used=0, paid_kickbacks=0, error=SimulationError.VALIDATION_ERROR, error_message=str(e))
            return self._convert_result_to_simulated_order(order, error_result)

    def simulate_order(self, order: Order) -> tuple[SimulatedOrder, HexBytes]:
        """
        Simulate an order, returning the simulation result and a state checkpoint.
        """
        self.vm.state.lock_changes()
        checkpoint = self.vm.state.snapshot()
        return self._simulate_single_order(order), checkpoint
    
    def get_max_blob_gas_per_block(self) -> int:
        vm_class_name = self.vm.__class__.__name__.lower()
        if "cancun" in vm_class_name:
            return CANCUN_MAX_BLOB_GAS
        elif "prague" in vm_class_name:
            return PRAGUE_MAX_BLOB_GAS
        else:
            return 0

    def _execute_tx(self, tx: SignedTransactionAPI, accumulated_trace=None) -> OrderSimResult:
        try:
            self._validate_tx(tx)

            coinbase_addr = self.context.coinbase
            initial_coinbase_balance = self.env.get_balance(coinbase_addr)
            prev_block_gas = self.header.gas_used

            # Start state tracing (patches EVM opcodes)
            self.state_tracer.start_tracing(tx)

            # Execute the transaction in the EVM
            computation = self.vm.state.apply_transaction(tx)  # low‑level exec
            
            receipt     = self.vm.make_receipt(self.header, tx, computation, self.vm.state)
            self.vm.validate_receipt(receipt)
            self.header = self.vm.add_receipt_to_header(self.header, receipt)

            self.header = self.vm.increment_blob_gas_used(self.header, tx)
            self.vm._initial_header = self.header  # Update the VM's header reference

            # Finish state tracing and get the trace
            tx_state_trace = self.state_tracer.finish_tracing(computation)
            
            if accumulated_trace is not None:
                final_trace = accumulated_trace.copy()
                final_trace.append_trace(tx_state_trace)
            else:
                final_trace = tx_state_trace

            # Clean up tracing (restore original opcodes)
            self.state_tracer.cleanup()

            final_coinbase_balance = self.env.get_balance(coinbase_addr)
            coinbase_profit = self._calculate_coinbase_profit(initial_coinbase_balance, final_coinbase_balance)

            return OrderSimResult(
                success=True,
                gas_used=receipt.gas_used - prev_block_gas,
                coinbase_profit=coinbase_profit,
                blob_gas_used=get_total_blob_gas(tx),
                paid_kickbacks=0,
                error=None,
                error_message=None,
                state_changes=None,
                state_trace=final_trace
            )

        except Exception as e:
            self.state_tracer.cleanup()
            tx_hash_str = tx.hash.hex() if tx is not None and hasattr(tx, "hash") else "unknown"
            logger.error(
                f"Transaction {tx_hash_str} simulation failed during validation: {str(e)}"
            )
            return OrderSimResult(
                success=False,
                gas_used=0,
                coinbase_profit=0,
                blob_gas_used=0,
                paid_kickbacks=0,
                error=SimulationError.VALIDATION_ERROR,
                error_message=str(e),
                state_trace=accumulated_trace
            )

    def _create_block_header(self) -> BlockHeaderAPI:
        """
        Build a fork-specific BlockHeader (Cancun, Prague, etc.), populated
        with our exact on-chain context values.
        """
        # Get every on-chain field from on-chain block context
        all_fields = {
            "block_number":             self.context.block_number,
            "timestamp":                self.context.block_timestamp,
            "base_fee_per_gas":         self.context.block_base_fee,
            "gas_limit":                self.context.block_gas_limit,
            "parent_hash":              self._safe_to_bytes(self.context.parent_hash),
            "uncles_hash":              self._safe_to_bytes(self.context.uncles_hash),
            "state_root":               self._safe_to_bytes(self.context.state_root),
            "transaction_root":         self._safe_to_bytes(self.context.transaction_root),
            "receipt_root":             self._safe_to_bytes(self.context.receipt_root),
            "difficulty":               self.context.block_difficulty,
            "gas_used":                 0,
            "coinbase":                 self._safe_to_bytes(self.context.coinbase),
            "nonce":                    self._safe_to_bytes(self.context.nonce),
            "mix_hash":                 self._safe_to_bytes(self.context.mix_hash),
            "extra_data":               self._safe_to_bytes(self.context.extra_data),
            "parent_beacon_block_root": self._safe_to_bytes(self.context.parent_beacon_block_root),
            "requests_hash":            self._safe_to_bytes(self.context.requests_hash),
            "withdrawals_root":         self._safe_to_bytes(self.context.withdrawals_root),
            "bloom":                    self._safe_to_int(self.context.bloom),
            "excess_blob_gas":          self.context.excess_blob_gas,
        }

        parent_header = self.header  # Current header is the parent, set when we forked
        # Let the VM build its fork‐specific header subclass
        self.header = self.vm.create_header_from_parent(parent_header)

        # Inject fields on this same subclass
        meta_fields = set(self.header._meta.field_names)
        remaining = (meta_fields - {"block_number"})
        fields_to_inject = {
            name: all_fields[name]
            for name in remaining
            if name in all_fields and all_fields[name] is not None
        }

        if fields_to_inject:
            self.header = self.header.copy(**fields_to_inject)

        self.vm._initial_header = self.header  # Update the VM's header reference

        return self.header

    def _override_execution_context(self):
        # Set execution context parameters based on the simulation context for the current block
        self.env.evm.vm.state.execution_context._base_fee_per_gas = self.context.block_base_fee
        self.env.evm.vm.state.execution_context._coinbase = self._safe_to_bytes(self.context.coinbase)
        self.env.evm.vm.state.execution_context._timestamp = self._safe_to_int(self.context.block_timestamp)
        self.env.evm.vm.state.execution_context._block_number = self._safe_to_int(self.context.block_number)
        self.env.evm.vm.state.execution_context._gas_limit = self._safe_to_int(self.context.block_gas_limit)
        self.env.evm.vm.state.execution_context._mix_hash = self._safe_to_bytes(self.context.mix_hash)
        
        if self.context.excess_blob_gas is not None:
            self.env.evm.vm.state.execution_context._excess_blob_gas = self._safe_to_int(self.context.excess_blob_gas)

        # prev_hashes and chain_id are handled by the patched titanoboa, so we do not set them here

    def _validate_upfront_cost(self, tx: SignedTransactionAPI):
        """ 
        Validate that the sender has enough balance to cover the transaction's upfront cost.
        REVM does this automatically, but py-evm does not.
        """
        fee_per_gas = tx.max_fee_per_gas if hasattr(tx, "max_fee_per_gas") else tx.gas_price
        upfront = tx.value + fee_per_gas * tx.gas
        balance = self.vm.state.get_balance(tx.sender)
        if balance < upfront:
            raise ValidationError(
                f"LackOfFundForMaxFee {{ fee: {upfront}, balance: {balance} }}"
            )
    
    def _validate_tx(self, tx: SignedTransactionAPI):
        self.vm.validate_transaction_against_header(self.header, tx)
        self._validate_upfront_cost(tx)

        # Check (tx blob gas + header blob gas) is below limit
        total_blob_gas = get_total_blob_gas(tx) + self.header.blob_gas_used
        max_blob_gas = self.get_max_blob_gas_per_block()
        if total_blob_gas > max_blob_gas:
            raise ValidationError(
                f"BlobGasExceeded {{ Tx blob gas: {total_blob_gas} > {max_blob_gas} }}"
            )


    def _convert_result_to_simulated_order(self, order: Order, result: OrderSimResult) -> SimulatedOrder:
        if result.success:
            sim_value = SimValue(
                coinbase_profit=result.coinbase_profit, gas_used=result.gas_used,
                blob_gas_used=result.blob_gas_used, paid_kickbacks=result.paid_kickbacks,
                mev_gas_price=result.coinbase_profit / result.gas_used if result.gas_used > 0 else Decimal(0)
            )
            return SimulatedOrder(order=order, sim_value=sim_value, used_state_trace=result.state_trace)
        else:
            sim_value = SimValue(coinbase_profit=0, gas_used=0, blob_gas_used=0, mev_gas_price=0, paid_kickbacks=0)
            simulated_order = SimulatedOrder(order=order, sim_value=sim_value, used_state_trace=result.state_trace)
            simulated_order._error_result = result
            return simulated_order

    def _calculate_coinbase_profit(self, initial_balance: int, final_balance: int) -> int:
        return max(0, final_balance - initial_balance)

    def _safe_to_int(self, value: int | str | bytes) -> int:
        if not value:
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, bytes):
            return int.from_bytes(value, byteorder='big')
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("b'") or value.startswith('b"'):
                try:
                    # Safely parse as Python bytes literal
                    value = ast.literal_eval(value)
                    return int.from_bytes(value, byteorder='big')
                except Exception as e:
                    raise ValueError(f"Failed to parse bytes string: {value}") from e
            if value.lower().startswith("0x"):
                return int(value, 16)
            return int(value)
        raise TypeError(f"Unsupported type for int conversion: {type(value)}")


    def _safe_to_bytes(self, value) -> bytes:
        if not value:
            return b''
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("0x"):
                return bytes.fromhex(value[2:])
            if value.startswith("b'") or value.startswith('b"'):
                try:
                    return ast.literal_eval(value)
                except Exception:
                    pass
            try:
                return value.encode('latin1')  # Fallback: encode raw string
            except Exception:
                pass
        return bytes(value)

    def _simulate_single_order(self, order: Order, accumulated_trace=None) -> SimulatedOrder:
        """
        Simulate a single order with an optional accumulated trace from parent orders.
        """
        if hasattr(order, 'order_type'):
            otype = order.order_type()
            if otype == OrderType.TX and isinstance(order, TxOrder):
                return self._simulate_tx_order(order, accumulated_trace)
            elif otype == OrderType.BUNDLE and isinstance(order, BundleOrder):
                return self._simulate_bundle_order(order, accumulated_trace)
            elif otype == OrderType.SHAREBUNDLE and hasattr(self, '_simulate_share_bundle_order'):
                return self._simulate_share_bundle_order(order, accumulated_trace)
            else:
                error_message = f"Unknown or unsupported order type: {otype}"
                logger.error(error_message)
                error_result = OrderSimResult(success=False, gas_used=0, error=SimulationError.UNKNOWN_ERROR, error_message=error_message)
                return self._convert_result_to_simulated_order(order, error_result)
        else:
            error_message = "Order does not have order_type method."
            logger.error(error_message)
            error_result = OrderSimResult(success=False, gas_used=0, error=SimulationError.UNKNOWN_ERROR, error_message=error_message)
            return self._convert_result_to_simulated_order(order, error_result)

    def _simulate_tx_order(self, order: TxOrder, accumulated_trace=None) -> SimulatedOrder:
        logger.debug(f"Simulating transaction {order.id()}")
        try:
            tx = order.get_vm_transaction()
            result = self._execute_tx(tx, accumulated_trace)
            logger.debug(f"Simulation result: {result.success}, gas used: {result.gas_used}, coinbase profit: {result.coinbase_profit}")
            return self._convert_result_to_simulated_order(order, result)
        except Exception as e:
            logger.error(f"Failed to simulate tx order {getattr(order, 'id', lambda: '?')()}: {e}")
            error_result = OrderSimResult(success=False, gas_used=0, error=SimulationError.VALIDATION_ERROR, error_message=str(e))
            return self._convert_result_to_simulated_order(order, error_result)

    def _simulate_bundle_order(self, order: BundleOrder, accumulated_trace=None) -> SimulatedOrder:
        total_gas, total_profit = 0, 0
        combined_trace = accumulated_trace.copy() if accumulated_trace else None
        
        for child_order, optional in order.child_orders:
            res = self._execute_tx(child_order.get_vm_transaction(), combined_trace)
            if not res.success and not optional:
                return self._convert_result_to_simulated_order(order, res)
            if res.success:
                total_gas += res.gas_used
                total_profit += res.coinbase_profit
                
                if res.state_trace:
                    if combined_trace is None:
                        combined_trace = res.state_trace
                    else:
                        # The trace from _execute_tx already includes accumulated trace
                        combined_trace = res.state_trace
        
        combined = OrderSimResult(
            success=True, 
            gas_used=total_gas, 
            coinbase_profit=total_profit, 
            blob_gas_used=0, 
            paid_kickbacks=0, 
            error=None, 
            error_message=None, 
            state_changes=None, 
            state_trace=combined_trace
        )
        return self._convert_result_to_simulated_order(order, combined)

    def _simulate_share_bundle_order(self, order: ShareBundleOrder, accumulated_trace=None) -> SimulatedOrder:
        return self._simulate_bundle_order(order, accumulated_trace)

def simulate_orders(orders: List[Order], simulator: EVMSimulator) -> List[SimulatedOrder]:
    """
    Simulate orders using the provided EVM simulator instance.
    
    Args:
        orders: List of orders to simulate
        simulator: EVM simulator instance to use for simulation
        
    Returns:
        List of simulated orders
    """
    try:

        # Get initial on-chain nonces once
        on_chain_nonces = {addr: simulator.env.evm.vm.state.get_nonce(Address(addr).canonical_address) 
                           for order in orders for addr in {n.address for n in order.nonces()}}

        sim_tree = SimTree(on_chain_nonces)
        for order in orders:
            sim_tree.push_order(order)
        
        sim_results_final: List[SimulatedOrder] = []
        
        # The main simulation loop
        while True:
            sim_requests = sim_tree.pop_simulation_requests(limit=100)
            if not sim_requests:
                break # No more ready orders to process

            for request in sim_requests:
                simulated_order = simulator.simulate_order_with_parents(request.order, request.parents)
                sim_results_final.append(simulated_order) # Add result regardless of success

                if simulated_order.simulation_result.success:
                    # Create a result object and submit it to the tree to wake up other orders
                    nonces_after = [NonceKey(n.address, n.nonce + 1) for n in simulated_order.order.nonces()]
                    result_to_submit = SimulatedResult(
                        simulated_order=simulated_order,
                        parents=request.parents,
                        nonces_after=nonces_after
                    )
                    sim_tree.submit_simulation_result(result_to_submit)
        
        # Handle any orders that are still pending. They are unresolvable.
        for order_id, (order, _) in sim_tree.pending_orders.items():
            logger.debug(f"Order {order_id} has unresolvable nonce dependencies.")
            error_result = OrderSimResult(
                success=False, gas_used=0, error=SimulationError.INVALID_NONCE,
                error_message="Unresolvable nonce dependencies"
            )
            sim_results_final.append(simulator._convert_result_to_simulated_order(order, error_result))

        # Log summary
        successful = sum(1 for r in sim_results_final if r.simulation_result.success)
        failed = len(sim_results_final) - successful
        logger.debug(f"Block {simulator.context.block_number} simulation completed: {successful} successful, {failed} failed")

        return sim_results_final

    except Exception as e:
        logger.error(f"Block simulation failed: {e}", exc_info=True)
        raise
