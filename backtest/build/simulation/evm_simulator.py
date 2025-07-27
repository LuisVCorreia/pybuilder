from decimal import Decimal
import logging
from typing import List, Optional
from hexbytes import HexBytes
import ast
import boa_ext

from boa.vm.py_evm import Address
from boa.rpc import EthereumRPC, to_hex
from boa.environment import Env
from eth.abc import SignedTransactionAPI
from eth.vm.forks.berlin.transactions import AccessListTransaction
from eth.vm.forks.london.transactions import DynamicFeeTransaction
from eth.vm.forks.cancun.transactions import BlobTransaction, CancunTypedTransaction, CancunLegacyTransaction
from eth.vm.forks.cancun.headers import CancunBlockHeader

from backtest.common.order import Order, OrderType, TxOrder, BundleOrder, ShareBundleOrder
from .sim_tree import SimTree, SimulatedResult, NonceKey
from .sim_utils import SimulationContext, SimulatedOrder, SimValue, OrderSimResult, SimulationError
from .pyevm_opcode_state_tracer import PyEVMOpcodeStateTracer
from .state_trace import UsedStateTrace

logger = logging.getLogger(__name__)

class EVMSimulator:
    def __init__(self, simulation_context: SimulationContext, rpc_url: str, fetch_prev_hashes: bool = True):
        self.context = simulation_context
        self.rpc = EthereumRPC(rpc_url)
        self.env = Env(fast_mode_enabled=True, fork_try_prefetch_state=True)
        self.rpc_url = rpc_url
        self.vm = self.env.evm.vm
        self.state_tracer = PyEVMOpcodeStateTracer(self.env)  
        self.fork_at_block(self.context.block_number - 1)

    def fork_at_block(self, block_number: int):
        # Forking the EVM will re-initiliase the correct VM based on the block timestamp
        block_id = to_hex(block_number)
        self.env.fork_rpc(self.rpc, block_identifier=block_id)
        self.vm = self.env.evm.vm
        self._override_execution_context()  # Ensure execution context is set correctly

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

    def _execute_tx(self, tx_data, accumulated_trace=None) -> OrderSimResult:
        try:
            tx = self._create_pyevm_tx(tx_data)
            header = self._create_block_header()

            coinbase_addr = self.context.coinbase
            initial_coinbase_balance = self.env.get_balance(coinbase_addr)

            # Start state tracing (patches EVM opcodes)
            self.state_tracer.start_tracing(tx)

            # Validate header & tx manually
            self.vm.validate_transaction_against_header(header, tx)

            # Execute the transaction in the EVM
            computation = self.vm.state.apply_transaction(tx)  # low‑level exec
            receipt     = self.vm.make_receipt(header, tx, computation, self.vm.state)
            self.vm.validate_receipt(receipt)

            # Finish state tracing and get the trace (pass tx for simple transfer handling)
            tx_state_trace = self.state_tracer.finish_tracing(computation, tx)
            
            if accumulated_trace is not None:
                final_trace = accumulated_trace.copy()
                final_trace.append_trace(tx_state_trace)
            else:
                final_trace = tx_state_trace

            # Clean up tracing (restore original opcodes)
            self.state_tracer.cleanup()

            final_coinbase_balance = self.env.get_balance(coinbase_addr)
            coinbase_profit = self._calculate_coinbase_profit(initial_coinbase_balance, final_coinbase_balance)

            if computation.is_error and receipt.gas_used == 0:
                # Note: If tx reverted but used gas, we don't treat it as an error
                self.state_tracer.cleanup()
                logger.error(f"Transaction 0x{tx.hash.hex()} simulation failed during validation: {str(e)}")
                return OrderSimResult(
                    success=False,
                    gas_used=0,
                    coinbase_profit=0,
                    blob_gas_used=0,
                    paid_kickbacks=0,
                    error=SimulationError.VALIDATION_ERROR,
                    error_message=str(e),
                    state_trace=accumulated_trace  # Return accumulated trace even on failure
                )


            return OrderSimResult(
                success=True,
                gas_used=receipt.gas_used,
                coinbase_profit=coinbase_profit,
                blob_gas_used=getattr(computation, 'blob_gas_used', 0),
                paid_kickbacks=0,
                error=None,
                error_message=None,
                state_changes=None,
                state_trace=final_trace
            )

        except Exception as e:
            self.state_tracer.cleanup()
            logger.error(f"Transaction 0x{tx.hash.hex()} simulation failed during validation: {str(e)}")
            return OrderSimResult(
                success=False,
                gas_used=0,
                coinbase_profit=0,
                blob_gas_used=0,
                paid_kickbacks=0,
                error=SimulationError.VALIDATION_ERROR,
                error_message=str(e),
                state_trace=accumulated_trace  # Return accumulated trace even on failure
            )


    def _create_pyevm_tx(self, tx_data: dict) -> SignedTransactionAPI:
        tx_type_int = self._safe_to_int(tx_data.get("type", 0))
        to_addr = Address(tx_data["to"]) if tx_data["to"] else None
        value = self._safe_to_int(tx_data["value"])
        gas = self._safe_to_int(tx_data["gas"])
        gas_price = self._safe_to_int(tx_data["gasPrice"])
        data = self._safe_to_bytes(tx_data["data"])
        access_list = tx_data.get("accessList", [])
        nonce = self._safe_to_int(tx_data["nonce"])

        v = self._safe_to_int(tx_data.get("v"))
        r = self._safe_to_int(tx_data.get("r"))
        s = self._safe_to_int(tx_data.get("s"))

        if tx_type_int == 1:
            y_parity = v & 1
            inner_tx = AccessListTransaction(
                chain_id=1, nonce=nonce, gas_price=gas_price, gas=gas, to=to_addr.canonical_address,
                value=value, data=data, access_list=access_list, y_parity=y_parity, r=r, s=s,
            )
            # Wrap in CancunTypedTransaction because you’re on the Cancun VM
            return CancunTypedTransaction(1, inner_tx)
        
        elif tx_type_int == 2:
            y_parity = self._safe_to_int(tx_data["y_parity"])
            max_fee_per_gas = self._safe_to_int(tx_data["max_fee_per_gas"])
            max_priority_fee_per_gas = self._safe_to_int(tx_data["max_priority_fee_per_gas"])
            inner_tx = DynamicFeeTransaction(
                chain_id=1, nonce=nonce, max_priority_fee_per_gas=max_priority_fee_per_gas,
                max_fee_per_gas=max_fee_per_gas, gas=gas, to=self._safe_to_bytes(to_addr.canonical_address) if to_addr else b'',
                value=value, data=data, access_list=access_list, y_parity=y_parity, r=r, s=s
            )
            return CancunTypedTransaction(2, inner_tx)
        elif tx_type_int == 3:
            y_parity = self._safe_to_int(tx_data["y_parity"])
            max_fee_per_gas = self._safe_to_int(tx_data["max_fee_per_gas"])
            max_priority_fee_per_gas = self._safe_to_int(tx_data["max_priority_fee_per_gas"])
            max_fee_per_blob_gas = self._safe_to_int(tx_data.get("max_fee_per_blob_gas"))
            blob_hashes = tx_data.get("blob_versioned_hashes")
            inner_tx = BlobTransaction(
                chain_id=1, nonce=nonce, max_priority_fee_per_gas=max_priority_fee_per_gas,
                max_fee_per_gas=max_fee_per_gas, gas=gas, to=to_addr.canonical_address,
                value=value, data=data, max_fee_per_blob_gas=max_fee_per_blob_gas,
                blob_versioned_hashes=blob_hashes, access_list=access_list, y_parity=y_parity, r=r, s=s,
            )
            return CancunTypedTransaction(3, inner_tx)
        else:
            return CancunLegacyTransaction(
                nonce=nonce, gas_price=gas_price, gas=gas, to=to_addr.canonical_address,
                value=value, data=data, v=v, r=r, s=s,
            )

    def _create_block_header(self) -> CancunBlockHeader:
        return CancunBlockHeader(
            block_number=self.context.block_number, timestamp=self.context.block_timestamp,
            base_fee_per_gas=self.context.block_base_fee, gas_limit=self.context.block_gas_limit,
            parent_hash=self.context.parent_hash, uncles_hash=self.context.uncles_hash,
            state_root=self.context.state_root, transaction_root=self.context.transaction_root,
            nonce=self.context.nonce, coinbase=self.context.coinbase,
            difficulty=self.context.block_difficulty, withdrawals_root=self.context.withdrawals_root,
            gas_used=0, excess_blob_gas=self.context.excess_blob_gas, receipt_root=self.context.receipt_root,
            mix_hash=self.context.mix_hash, parent_beacon_block_root=self.context.parent_beacon_block_root,
            bloom=self.context.bloom, extra_data=self.context.extra_data
        )

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

        # prev_hashes and chain_id are handled by the patched titanoboa, so we do not set them here (this enables caching)

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
            value = value.strip().lower()
            if value == "0x" or value == "":
                return 0
            return int(value, 16)
        raise TypeError(f"Unsupported type for int conversion: {type(value)}")


    def _safe_to_bytes(self, value) -> bytes:
        if value is None:
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
            tx_data = order.get_transaction_data()
            result = self._execute_tx(tx_data, accumulated_trace)
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
            res = self._execute_tx(child_order.get_transaction_data(), combined_trace)
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

        # 1. Get initial on-chain nonces once
        on_chain_nonces = {addr: simulator.env.evm.vm.state.get_nonce(Address(addr).canonical_address) 
                           for order in orders for addr in {n.address for n in order.nonces()}}

        sim_tree = SimTree(on_chain_nonces)
        for order in orders:
            sim_tree.push_order(order)
        
        sim_results_final: List[SimulatedOrder] = []
        
        # 2. The main simulation loop
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
        
        # 3. Handle any orders that are still pending. They are unresolvable.
        for order_id, (order, _) in sim_tree.pending_orders.items():
            logger.warning(f"Order {order_id} has unresolvable nonce dependencies.")
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

# def simulate_orders(orders: List[Order], simulator: EVMSimulator) -> List[SimulatedOrder]:


#     # Execute first tx with hash tx:0x95c572ec5a475c615cf24758c367e834df214119139e6e573c713156d99325ad
#     # then tx with hash tx:0x0de7a57fb278beca6c802e2e007c29df311127080e2e37a189a4a8702cf567c6

#     # get the first order
#     first_order = next((o for o in orders if o.id().value == "0xb05a3f2b8891db2ac45508cc1fb3d8504478b2f37059b970d0ba3d9af63bc223"), None)
#     second_order = next((o for o in orders if o.id().value == "0x3c41a592347a917d6e2d72bf0cf47e2902d5ec4053c4d8cbe0505de8799a579c"), None)
#     # third_order = next((o for o in orders if o.id().value == "0x1912bb377d6a2e13c76b3a48ef6c5b793eda305943cdde0f685abe3a851d6b88"), None)
#     # if not first_order or not second_order or not third_order:
#     #     raise ValueError("Required orders not found in the provided list")
    

#     # Simulate first order
#     first_simulated_order = simulator._simulate_tx_order(first_order)
#     logger.info(f"First order simulation result: {first_simulated_order.simulation_result.success}, gas used: {first_simulated_order.simulation_result.gas_used}, coinbase profit: {first_simulated_order.simulation_result.coinbase_profit}")

#     # # Simulate second order
#     # second_simulated_order = simulator._simulate_tx_order(second_order)
#     # logger.info(f"Second order simulation result: {second_simulated_order.simulation_result.success}, gas used: {second_simulated_order.simulation_result.gas_used}, coinbase profit: {second_simulated_order.simulation_result.coinbase_profit}")

#     # # Simulate third order
#     # third_simulated_order = simulator.simulate_tx_order(third_order)
#     # logger.info(f"Third order simulation result: {third_simulated_order.simulation_result.success}, gas used: {third_simulated_order.simulation_result.gas_used}, coinbase profit: {third_simulated_order.simulation_result.coinbase_profit}")