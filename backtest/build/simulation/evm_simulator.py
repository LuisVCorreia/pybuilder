import logging
from backtest.common.block_data import BlockData
from backtest.common.order import Order, OrderType, TxOrder, BundleOrder, ShareBundleOrder
from typing import List
import web3
from .sim_tree import NonceKey, SimTree, SimulatedResult
from .sim_utils import SimulationContext, SimulatedOrder, SimValue, OrderSimResult, SimulationError
from .state_trace import UsedStateTrace
from .state_tracer import TracingEVMWrapper, create_tracing_evm
from eth_abi import decode

from boa.vm.py_evm import Address
from boa.rpc import EthereumRPC, to_hex, to_int, to_bytes
from boa.environment import Env

from eth.vm.forks.london.transactions import DynamicFeeTransaction
from eth.vm.forks.cancun.transactions import BlobTransaction
from eth.vm.forks.cancun.transactions import CancunTypedTransaction
from eth.vm.forks.cancun.transactions import CancunLegacyTransaction
from eth.vm.forks.cancun.headers import CancunBlockHeader

logger = logging.getLogger(__name__)


class EVMSimulator_pyEVM:
    def __init__(self,simulation_context: SimulationContext, rpc_url: str):
        self.context = simulation_context
        self.rpc = EthereumRPC(rpc_url)
        self.env = Env(fast_mode_enabled=True, fork_try_prefetch_state=True)    # Creates new environment with Py-EVM execution and remote RPC support
        self.vm = self.env.evm.vm
        self.rpc_url = rpc_url
        self.recent_block_hashes = []  # Store recent block hashes for BLOCKHASH opcode
        self.fork_at_block(self.context.block_number - 1)  # Fork at the parent block to simulate correctly
        
        # Fetch the previous 256 block hashes for BLOCKHASH opcode support
        self._fetch_recent_block_hashes(self.context.block_number)

    def fork_at_block(self, block_number: int):
        """Fork the EVM state at the specified block number and fetch recent block hashes."""
        block_id = to_hex(block_number)       
        # Use the env's fork_rpc method instead of direct EVM access
        self.env.fork_rpc(self.rpc, block_identifier=block_id)

    
    def _fetch_recent_block_hashes(self, current_block: int):
        """Fetch the previous 255 block hashes and store them for BLOCKHASH opcode."""
        w3 = web3.Web3(web3.Web3.HTTPProvider(self.rpc_url))
        
        logger.info(f"Fetching recent block hashes for block {current_block}")
        
        # Fetch hashes for blocks [current_block - 255, current_block - 1]
        # We already have the parent hash in our context
        start_block = max(0, current_block - 255)
        
        block_hashes = []
        
        if current_block > 0:
            block_hashes.append(self._safe_to_bytes(self.context.parent_hash))
        
        for block_num in range(current_block - 2, start_block - 1, -1):
            if block_num < 0:
                break
            try:
                block = w3.eth.get_block(block_num)
                block_hashes.append(self._safe_to_bytes(block['hash']))
            except Exception as e:
                logger.warning(f"Failed to fetch block hash for block {block_num}: {e}")
                # Set to zero hash if we can't fetch it
                block_hashes.append(b'\x00' * 32)
        
        self.recent_block_hashes = block_hashes
        logger.info(f"Fetched {len(self.recent_block_hashes)} block hashes")

    def override_execution_context(self):
        self.env.evm.vm.state.execution_context._base_fee_per_gas = self.context.block_base_fee
        self.env.evm.vm.state.execution_context._coinbase = self._safe_to_bytes(self.context.coinbase)

        # Set the context to the block where the transaction is being included
        self.env.evm.vm.state.execution_context._timestamp = self._safe_to_int(self.context.block_timestamp)
        self.env.evm.vm.state.execution_context._block_number = self._safe_to_int(self.context.block_number)
        self.env.evm.vm.state.execution_context._gas_limit = self._safe_to_int(self.context.block_gas_limit)

        self.env.evm.vm.state.execution_context._base_fee_per_gas = self._safe_to_int(self.context.block_base_fee)
        self.env.evm.vm.state.execution_context._mix_hash = self._safe_to_bytes(self.context.mix_hash)
        self.env.evm.vm.state.execution_context._excess_blob_gas = self._safe_to_bytes(self.context.excess_blob_gas)
        
        # Set recent block hashes for BLOCKHASH opcode support
        self.env.evm.vm.state.execution_context._prev_hashes = self.recent_block_hashes

    def _safe_to_hex(self, value: int | bytes | str) -> str:
        if isinstance(value, int):
            return hex(value)
        if isinstance(value, bytes):
            return "0x" + value.hex()
        if isinstance(value, str):
            assert value.startswith("0x")
            return value
        raise TypeError(
            f"to_hex expects bytes, int or (hex) string, but got {type(value)}: {value}"
        )

    def _safe_to_int(self, value: int | str | bytes) -> int:
        """Convert a value to int, handling hex strings, integers, and bytes."""
        if not value:
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, bytes):
            return int.from_bytes(value, byteorder='big')
        if value == "0x":
            return 0
        return int(value, 16)

    def _safe_to_bytes(self, value) -> bytes:
        """Convert a value to bytes, handling both hex strings and bytes."""
        if isinstance(value, bytes):
            return value
        elif isinstance(value, str):
            # Handle string representation of bytes
            if value.startswith("b'") and value.endswith("'"):
                # This is a string representation of bytes, evaluate it
                try:
                    return eval(value)
                except:
                    pass
            # Handle regular hex strings
            return bytes.fromhex(value.removeprefix("0x"))
        else:
            return bytes(value)
    
        # Add this method to your EVMSimulator class
    def _decode_revert_output(self, output_hex: str) -> str:
        """Decode revert output to human-readable error message"""
        return self.decode_revert_reason(output_hex)

    def decode_revert_reason(self, revert_data: str) -> str:
        """
        Decode ABI-encoded revert reason from transaction failure.
        
        Args:
            revert_data: Hex string of revert data (with or without 0x prefix)
            
        Returns:
            Decoded error message string
        """
        if not revert_data:
            return "No revert reason provided"
            
        # Remove 0x prefix if present
        if revert_data.startswith('0x'):
            revert_data = revert_data[2:]
        
        try:
            # Convert hex to bytes

            data = bytes.fromhex(revert_data)
            
            # Check if it's a standard Error(string) revert (selector 0x08c379a0)
            if len(data) >= 4 and data[:4] == bytes.fromhex('08c379a0'):
                # Decode the string parameter
                decoded = decode(['string'], data[4:])
                return decoded[0]
            else:
                # Try to decode as raw string if not standard format
                try:
                    return data.decode('utf-8').strip('\x00')
                except:
                    return f"Custom error or non-string revert: {revert_data}"
                    
        except Exception as e:
            return f"Failed to decode revert reason: {e}"

    def _execute_tx(self, tx_data) -> OrderSimResult:
        """Simulate a transaction execution."""
        # Extract transaction parameters
        tx_type = tx_data["type"]
        tx_type_int = self._safe_to_int(tx_type) if tx_type else 0
        to_addr = Address(tx_data["to"]) if tx_data["to"] else None
        value = self._safe_to_int(tx_data["value"])
        gas = self._safe_to_int(tx_data["gas"])
        gas_price = self._safe_to_int(tx_data["gasPrice"])
        data = self._safe_to_bytes(tx_data["data"])
        access_list = tx_data.get("accessList", [])
        nonce = self._safe_to_int(tx_data["nonce"])
        max_fee_per_blob_gas = None
        blob_hashes = None

        v = self._safe_to_int(tx_data.get("v"))
        r = self._safe_to_int(tx_data.get("r"))
        s = self._safe_to_int(tx_data.get("s"))

        if tx_type_int == 2:  # EIP-1559 transaction
            y_parity = self._safe_to_int(tx_data["y_parity"])
            max_fee_per_gas = self._safe_to_int(tx_data["max_fee_per_gas"])
            max_priority_fee_per_gas = self._safe_to_int(tx_data["max_priority_fee_per_gas"])
        elif tx_type_int == 3:
            max_fee_per_blob_gas = self._safe_to_int(tx_data.get("max_fee_per_blob_gas"))
            blob_hashes = tx_data.get("blob_versioned_hashes")

        try:
            if tx_type_int == 2:         
                # Create Type 2 transaction object
                inner_tx = DynamicFeeTransaction(
                    chain_id=1,  # Mainnet
                    nonce=nonce,
                    max_priority_fee_per_gas=max_priority_fee_per_gas,
                    max_fee_per_gas=max_fee_per_gas,
                    gas=gas,
                    to=self._safe_to_bytes(to_addr.canonical_address) if to_addr else b'',
                    value=value,
                    data=data,
                    access_list=access_list,
                    y_parity=y_parity,
                    r=r,
                    s=s,
                )
                
                tx = CancunTypedTransaction(2, inner_tx)

            elif tx_type_int == 3:  # EIP-4844 transaction
                inner_tx = BlobTransaction(
                    chain_id=1,  # Mainnet
                    nonce=nonce,
                    max_priority_fee_per_gas=max_priority_fee_per_gas,
                    max_fee_per_gas=max_fee_per_gas,
                    gas=gas,
                    to=to_addr.canonical_address if to_addr else b'',
                    value=value,
                    data=data,
                    max_fee_per_blob_gas=max_fee_per_blob_gas,
                    blob_versioned_hashes=blob_hashes,
                    access_list=access_list,
                    y_parity=y_parity,
                    r=r,
                    s=s,
                )

                tx = CancunTypedTransaction(3, inner_tx)
                
            else:  # Legacy transaction                    
                tx = CancunLegacyTransaction(
                    nonce=nonce,
                    gas_price=gas_price,
                    gas=gas,
                    to=to_addr.canonical_address if to_addr else b'',
                    value=value,
                    data=data,
                    v=v,
                    r=r,
                    s=s,
                )
                        
            # Create proper block header using simulation context
            header = CancunBlockHeader(
                block_number=self.context.block_number,
                timestamp=self.context.block_timestamp,
                base_fee_per_gas=self.context.block_base_fee,
                gas_limit=self.context.block_gas_limit,
                parent_hash=self.context.parent_hash,
                uncles_hash=self.context.uncles_hash,
                state_root=self.context.state_root,
                transaction_root=self.context.transaction_root,
                nonce=self.context.nonce,
                coinbase=self.context.coinbase,
                difficulty=self.context.block_difficulty,
                withdrawals_root=self.context.withdrawals_root,
                gas_used=0,
                excess_blob_gas=self.context.excess_blob_gas,
                receipt_root=self.context.receipt_root,
                mix_hash=self.context.mix_hash,
                parent_beacon_block_root=self.context.parent_beacon_block_root,
            )

            coinbase_addr = self.context.coinbase
            initial_coinbase_balance = self.env.get_balance(coinbase_addr)

            self.override_execution_context()  # Ensure execution context is set correctly

            # Apply the transaction with the header
            receipt, computation = self.env.evm.vm.apply_transaction(header, tx)

            final_coinbase_balance = self.env.get_balance(coinbase_addr)
            coinbase_profit = self._calculate_coinbase_profit(initial_coinbase_balance, final_coinbase_balance)

            return OrderSimResult(
                success=True,
                gas_used=receipt.gas_used,
                coinbase_profit=coinbase_profit,
                blob_gas_used=getattr(computation, 'blob_gas_used', 0),
                paid_kickbacks=0,
                error=None,
                error_message=None,
                state_changes=None,
                state_trace=None
            )

        except Exception as e:
            # # Finish tracing even on error
            # if self.enable_tracing and hasattr(self.evm, 'finish_tracing'):
            #     try:
            #         state_trace = self.evm.finish_tracing()
            #     except:
            #         state_trace = None
            
            # # Check if the transaction used gas before failing
            # gas_used = getattr(self.evm.result, 'gas_used', 0) if hasattr(self.evm, 'result') else 0
            

            # # Try to decode revert reason if available
            # error_message = str(e)
            # # Extract hex output from the error message using regex
            # # Look for pattern like "output: 0x[hex_string]"
            # hex_match = re.search(r'output: (0x[0-9a-fA-F]+)', error_message)
            # if hex_match:
            #     hex_output = hex_match.group(1)
            #     decoded_reason = self._decode_revert_output(hex_output)
            #     error_message = f"{error_message} - Decoded: {decoded_reason}"
            
                
            # if gas_used > 0:
            #     # Transaction failed after using gas - builder can still collect fees
            #     profit = self.evm.get_balance(cb) - before_cb - self.context.block_base_fee * gas_used
            #     self.evm.set_balance(cb, before_cb + profit)
            #     # self.evm.commit()
                
            #     logger.info(f"Transaction reverted but used gas ({gas_used}), treating as successful: {error_message}")
            #     return OrderSimResult(
            #         success=True,
            #         gas_used=gas_used,
            #         coinbase_profit=profit,
            #         blob_gas_used=getattr(self.evm.result, 'blob_gas_used', 0) if hasattr(self.evm, 'result') else 0,
            #         paid_kickbacks=0,
            #         error=None,
            #         error_message=error_message,
            #         state_changes=None,
            #         state_trace=state_trace
            #     )
            # else:
            # Transaction failed before using gas - validation error
            logger.error(f"Transaction simulation failed during validation: {str(e)}")
            return OrderSimResult(
                success=False,
                gas_used=0,
                coinbase_profit=0,
                blob_gas_used=0,
                paid_kickbacks=0,
                error=SimulationError.VALIDATION_ERROR,
                error_message=str(e),
                state_trace=None
            )


    def simulate_tx_order(self, order: TxOrder) -> SimulatedOrder:
        logger.info(f"Simulating transaction {order.id()}")

        # Only accept TxOrder
        if not isinstance(order, TxOrder):
            logger.error(f"simulate_tx_order called with non-TxOrder: {type(order)}")
            error_result = OrderSimResult(
                success=False,
                gas_used=0,
                error=SimulationError.UNKNOWN_ERROR,
                error_message="simulate_tx_order called with non-TxOrder"
            )
            return self._convert_result_to_simulated_order(order, error_result)
        try:
            tx_data = order.get_transaction_data()
            result = self._execute_tx(tx_data)
            logger.info(f"Simulation result: {result.success}, gas used: {result.gas_used}, coinbase profit: {result.coinbase_profit}")
            return self._convert_result_to_simulated_order(order, result)
        except Exception as e:
            logger.error(f"Failed to simulate tx order {getattr(order, 'id', lambda: '?')()}: {e}")
            error_result = OrderSimResult(
            success=False,
            gas_used=0,
            error=SimulationError.VALIDATION_ERROR,
            error_message=str(e)
            )
            return self._convert_result_to_simulated_order(order, error_result)
  
    def simulate_bundle_order(self, order: BundleOrder) -> SimulatedOrder:
        # rollback = self.evm.snapshot()
        total_gas, total_profit = 0, 0
        for child_order, optional in order.child_orders:
            res = self._execute_tx(child_order.get_transaction_data())
            if not res.success and not optional:
                # self.evm.revert(rollback)
                return self._convert_result_to_simulated_order(order, res)
            if res.success:
                total_gas += res.gas_used
                total_profit += res.coinbase_profit
        # self.evm.revert(rollback)
        combined = OrderSimResult(success=True, gas_used=total_gas, coinbase_profit=total_profit)
        return self._convert_result_to_simulated_order(order, combined)
    
    def simulate_share_bundle_order(self, order: ShareBundleOrder) -> SimulatedOrder:
        # Only accept ShareBundleOrder
        if not isinstance(order, ShareBundleOrder):
            logger.error(f"simulate_share_bundle_order called with non-ShareBundleOrder: {type(order)}")
            error_result = OrderSimResult(
                success=False,
                gas_used=0,
                error=SimulationError.UNKNOWN_ERROR,
                error_message="simulate_share_bundle_order called with non-ShareBundleOrder"
            )
            return self._convert_result_to_simulated_order(order, error_result)
        return self.simulate_bundle_order(order)  # Only call with correct type
    
    def simulate_order_with_parents(self, order: Order, parent_orders: List[Order] = None) -> SimulatedOrder:
        """
        Simulate an order, including any required parent orders first.
        Maintains nonce tracking by recording each successful parent execution.
        """
        if parent_orders is None:
            parent_orders = []

        try:
            self.fork_at_block(self.context.block_number - 1)
            # First simulate all parent orders, updating nonce cache
            # checkpoint = self.evm.snapshot()
            for parent_order in parent_orders:
                parent_res = self._simulate_single_order(parent_order)
                if not parent_res.simulation_result.success:
                    error_result = OrderSimResult(
                        success=False,
                        gas_used=0,
                        coinbase_profit=0,
                        blob_gas_used=0,
                        paid_kickbacks=0,
                        error=SimulationError.VALIDATION_ERROR,
                        error_message=f"Parent order failed: {parent_res.simulation_result.error_message}"
                    )
                    # self.evm.revert(checkpoint)
                    return self._convert_result_to_simulated_order(order, error_result)

                # state_changes = self.evm.journal_state
                # self.evm = self._create_evm()  # Reset EVM for next parent order
                # for address, account_info in state_changes.items():
                #     # Insert account info (nonce, balance, codeHash)
                #     self.evm.insert_account_info(address, account_info)

                # self.evm.commit()
            # Now simulate main order
            main_res = self._simulate_single_order(order)
            # self.evm.revert(checkpoint)  # Revert to initial state after simulating parents
            return main_res
        except Exception as e:
            # On exception, rollback and return error
            # self.evm.revert(checkpoint)
            error_result = OrderSimResult(
                success=False,
                gas_used=0,
                coinbase_profit=0,
                blob_gas_used=0,
                paid_kickbacks=0,
                error=SimulationError.VALIDATION_ERROR,
                error_message=str(e)
            )
            return self._convert_result_to_simulated_order(order, error_result)
    
    def _simulate_single_order(self, order: Order) -> SimulatedOrder:
        """Simulate a single order without parent handling"""
        if hasattr(order, 'order_type'):
            otype = order.order_type()
            if otype == OrderType.TX and isinstance(order, TxOrder):
                return self.simulate_tx_order(order)
            elif otype == OrderType.BUNDLE and isinstance(order, BundleOrder):
                return self.simulate_bundle_order(order)
            elif otype == OrderType.SHAREBUNDLE and hasattr(self, 'simulate_share_bundle_order'):
                # If py-evm requires BundleOrder, try to convert or skip
                if hasattr(order, 'to_bundle_order'):
                    bundle_order = order.to_bundle_order()
                    return self.simulate_bundle_order(bundle_order)
                else:
                    logger.warning("ShareBundleOrder simulation not supported, skipping.")
                    error_result = OrderSimResult(
                        success=False,
                        gas_used=0,
                        error=SimulationError.UNKNOWN_ERROR,
                        error_message="ShareBundleOrder simulation not supported"
                    )
                    return self._convert_result_to_simulated_order(order, error_result)
            else:
                logger.error(f"Unknown or unsupported order type: {otype}")
                error_result = OrderSimResult(
                    success=False,
                    gas_used=0,
                    error=SimulationError.UNKNOWN_ERROR,
                    error_message=f"Order type {otype} not supported"
                )
                return self._convert_result_to_simulated_order(order, error_result)
        else:
            logger.error("Order does not have order_type method.")
            error_result = OrderSimResult(
                success=False,
                gas_used=0,
                error=SimulationError.UNKNOWN_ERROR,
                error_message="Order does not have order_type method."
            )
            return self._convert_result_to_simulated_order(order, error_result)

    def _convert_result_to_simulated_order(self, order: Order, result: OrderSimResult) -> SimulatedOrder:
        """Convert OrderSimResult to SimulatedOrder to match rbuilder's structure"""
        if result.success:
            sim_value = SimValue(
                coinbase_profit=result.coinbase_profit,
                gas_used=result.gas_used,
                blob_gas_used=result.blob_gas_used,
                paid_kickbacks=result.paid_kickbacks
            )
            return SimulatedOrder(
                order=order,
                sim_value=sim_value,
                used_state_trace=result.state_trace  # Include state trace
            )
        else:
            # For failed orders, create a SimulatedOrder with zero values
            sim_value = SimValue(
                coinbase_profit=0,
                gas_used=0,
                blob_gas_used=0,
                paid_kickbacks=0
            )
            simulated_order = SimulatedOrder(
                order=order,
                sim_value=sim_value,
                used_state_trace=result.state_trace  # Include state trace even for failed orders
            )
            # Store error info for backwards compatibility
            simulated_order._error_result = result
            return simulated_order

    def _calculate_coinbase_profit(self, initial_balance: int, final_balance: int) -> int:
        """Calculate coinbase profit from balance change"""
        return max(0, final_balance - initial_balance)

def simulate_orders(orders: List[Order], block_data: BlockData, rpc_url: str, enable_tracing: bool = True) -> List[SimulatedOrder]:
    try:
        context = SimulationContext.from_onchain_block(block_data.onchain_block)
        simulator = EVMSimulator_pyEVM(context, rpc_url)

        # 1. Get initial on-chain nonces once.
        # on_chain_nonces = {addr: simulator.evm.basic(addr).nonce for order in orders for addr in {n.address for n in order.nonces()}}
        on_chain_nonces = {addr: simulator.env.evm.vm.state.get_nonce(Address(addr).canonical_address) for order in orders for addr in {n.address for n in order.nonces()}}

        sim_tree = SimTree(on_chain_nonces)
        for order in orders:
            sim_tree.push_order(order)
        
        sim_results_final: List[SimulatedOrder] = []
        
        # 2. The main simulation loop.
        while True:
            sim_requests = sim_tree.pop_simulation_requests(limit=100)
            if not sim_requests:
                break # No more ready orders to process.

            for request in sim_requests:
                if request.parents and len(request.parents) > 1:
                    logger.info(f"Order {request.order.id()} has {len(request.parents)} parent orders")
                    # Save order id to a file for debugging
                    with open("sim_debug.txt", "a") as f:
                        f.write(f"Order {request.order.id()} has {len(request.parents)} parents\n")

                simulated_order = simulator.simulate_order_with_parents(request.order, request.parents)
                sim_results_final.append(simulated_order) # Add result regardless of success

                if simulated_order.simulation_result.success:
                    # Create a result object and submit it to the tree to wake up other orders.
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
        logger.info(f"Block {context.block_number} simulation completed: {successful} successful, {failed} failed")

        return sim_results_final

    except Exception as e:
        logger.error(f"Block simulation failed: {e}", exc_info=True)
        raise


# def simulate_orders(orders: List[Order], block_data: BlockData, rpc_url: str, enable_tracing: bool = True) -> List[SimulatedOrder]:
#     # Debugging version, we're going to simulate two orders in the same context
#     context = SimulationContext.from_onchain_block(block_data.onchain_block)
#     simulator = EVMSimulator_pyEVM(context, rpc_url)

#     # Execute first tx with hash tx:0x95c572ec5a475c615cf24758c367e834df214119139e6e573c713156d99325ad
#     # then tx with hash tx:0x0de7a57fb278beca6c802e2e007c29df311127080e2e37a189a4a8702cf567c6

#     # get the first order
#     first_order = next((o for o in orders if o.id().value == "0x6649cb24f5aec6971680fb63def402eae8340c52b7001aeaa79860a3a77d048f"), None)
#     second_order = next((o for o in orders if o.id().value == "0x3aed99d18cecd864e4170294f1c62d42c62d318ce16d1140be50738033d55a77"), None)
#     third_order = next((o for o in orders if o.id().value == "0x1912bb377d6a2e13c76b3a48ef6c5b793eda305943cdde0f685abe3a851d6b88"), None)
#     if not first_order or not second_order or not third_order:
#         raise ValueError("Required orders not found in the provided list")
    

#     # Simulate first order
#     first_simulated_order = simulator.simulate_tx_order(first_order)
#     logger.info(f"First order simulation result: {first_simulated_order.simulation_result.success}, gas used: {first_simulated_order.simulation_result.gas_used}, coinbase profit: {first_simulated_order.simulation_result.coinbase_profit}")

#     # state_changes = simulator.evm.journal_state
#     # sender_address = first_order.get_transaction_data()["from"]

#     # cold_simulator = EVMSimulator(context, rpc_url, enable_tracing=enable_tracing)
#     # for address, account_info in state_changes.items():
#     #     # Insert account info (nonce, balance, codeHash)
#     #     cold_simulator.evm.insert_account_info(address, account_info)

#     # # Simulate second order
#     # second_simulated_order = simulator.simulate_tx_order(second_order)
#     # logger.info(f"Second order simulation result: {second_simulated_order.simulation_result.success}, gas used: {second_simulated_order.simulation_result.gas_used}, coinbase profit: {second_simulated_order.simulation_result.coinbase_profit}")


#     # # Simulate third order
#     # third_simulated_order = simulator.simulate_tx_order(third_order)
#     # logger.info(f"Third order simulation result: {third_simulated_order.simulation_result.success}, gas used: {third_simulated_order.simulation_result.gas_used}, coinbase profit: {third_simulated_order.simulation_result.coinbase_profit}")