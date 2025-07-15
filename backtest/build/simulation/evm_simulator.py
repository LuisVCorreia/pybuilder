import logging
from typing import List
import web3
from eth_abi import decode

from boa.vm.py_evm import Address
from boa.rpc import EthereumRPC, to_hex
from boa.environment import Env
from eth.abc import SignedTransactionAPI
from eth.vm.forks.london.transactions import DynamicFeeTransaction
from eth.vm.forks.cancun.transactions import BlobTransaction, CancunTypedTransaction, CancunLegacyTransaction
from eth.vm.forks.cancun.headers import CancunBlockHeader

from backtest.common.order import Order, OrderType, TxOrder, BundleOrder, ShareBundleOrder
from .sim_tree import SimTree, SimulatedResult, NonceKey
from .sim_utils import SimulationContext, SimulatedOrder, SimValue, OrderSimResult, SimulationError

logger = logging.getLogger(__name__)


class EVMSimulator:
    def __init__(self, simulation_context: SimulationContext, rpc_url: str):
        self.context = simulation_context
        self.rpc = EthereumRPC(rpc_url)
        self.env = Env(fast_mode_enabled=True, fork_try_prefetch_state=True)
        self.vm = self.env.evm.vm
        self.rpc_url = rpc_url
        self.recent_block_hashes: List[bytes] = []
        self.fork_at_block(self.context.block_number - 1)
        self._fetch_recent_block_hashes(self.context.block_number)

    def simulate_tx_order(self, order: TxOrder) -> SimulatedOrder:
        logger.info(f"Simulating transaction {order.id()}")
        if not isinstance(order, TxOrder):
            logger.error(f"simulate_tx_order called with non-TxOrder: {type(order)}")
            error_result = OrderSimResult(success=False, gas_used=0, error=SimulationError.UNKNOWN_ERROR, error_message="simulate_tx_order called with non-TxOrder")
            return self._convert_result_to_simulated_order(order, error_result)
        try:
            tx_data = order.get_transaction_data()
            result = self._execute_tx(tx_data)
            logger.info(f"Simulation result: {result.success}, gas used: {result.gas_used}, coinbase profit: {result.coinbase_profit}")
            return self._convert_result_to_simulated_order(order, result)
        except Exception as e:
            logger.error(f"Failed to simulate tx order {getattr(order, 'id', lambda: '?')()}: {e}")
            error_result = OrderSimResult(success=False, gas_used=0, error=SimulationError.VALIDATION_ERROR, error_message=str(e))
            return self._convert_result_to_simulated_order(order, error_result)

    def simulate_bundle_order(self, order: BundleOrder) -> SimulatedOrder:
        total_gas, total_profit = 0, 0
        for child_order, optional in order.child_orders:
            res = self._execute_tx(child_order.get_transaction_data())
            if not res.success and not optional:
                return self._convert_result_to_simulated_order(order, res)
            if res.success:
                total_gas += res.gas_used
                total_profit += res.coinbase_profit
        combined = OrderSimResult(success=True, gas_used=total_gas, coinbase_profit=total_profit, blob_gas_used=0, paid_kickbacks=0, error=None, error_message=None, state_changes=None, state_trace=None)
        return self._convert_result_to_simulated_order(order, combined)

    def simulate_share_bundle_order(self, order: ShareBundleOrder) -> SimulatedOrder:
        if not isinstance(order, ShareBundleOrder):
            logger.error(f"simulate_share_bundle_order called with non-ShareBundleOrder: {type(order)}")
            error_result = OrderSimResult(success=False, gas_used=0, error=SimulationError.UNKNOWN_ERROR, error_message="simulate_share_bundle_order called with non-ShareBundleOrder")
            return self._convert_result_to_simulated_order(order, error_result)
        return self.simulate_bundle_order(order)

    def simulate_order_with_parents(self, order: Order, parent_orders: List[Order] = None) -> SimulatedOrder:
        parent_orders = parent_orders or []
        try:
            self.fork_at_block(self.context.block_number - 1)
            for parent_order in parent_orders:
                parent_res = self._simulate_single_order(parent_order)
                if not parent_res.simulation_result.success:
                    error_message = f"Parent order failed: {parent_res.simulation_result.error_message}"
                    error_result = OrderSimResult(success=False, gas_used=0, coinbase_profit=0, blob_gas_used=0, paid_kickbacks=0, error=SimulationError.VALIDATION_ERROR, error_message=error_message)
                    return self._convert_result_to_simulated_order(order, error_result)
            return self._simulate_single_order(order)
        except Exception as e:
            error_result = OrderSimResult(success=False, gas_used=0, coinbase_profit=0, blob_gas_used=0, paid_kickbacks=0, error=SimulationError.VALIDATION_ERROR, error_message=str(e))
            return self._convert_result_to_simulated_order(order, error_result)

    def simulate_and_commit_order(self, order: Order) -> SimulatedOrder:
        return self._simulate_single_order(order)

    def _simulate_single_order(self, order: Order) -> SimulatedOrder:
        if hasattr(order, 'order_type'):
            otype = order.order_type()
            if otype == OrderType.TX and isinstance(order, TxOrder):
                return self.simulate_tx_order(order)
            elif otype == OrderType.BUNDLE and isinstance(order, BundleOrder):
                return self.simulate_bundle_order(order)
            elif otype == OrderType.SHAREBUNDLE and hasattr(self, 'simulate_share_bundle_order'):
                return self.simulate_share_bundle_order(order)
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

    def _execute_tx(self, tx_data) -> OrderSimResult:
        """Simulate a transaction execution."""
        try:
            tx = self._create_pyevm_tx(tx_data)
            header = self._create_block_header()

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
        
        if tx_type_int == 2:
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
                max_fee_per_gas=max_fee_per_gas, gas=gas, to=to_addr.canonical_address if to_addr else b'',
                value=value, data=data, max_fee_per_blob_gas=max_fee_per_blob_gas,
                blob_versioned_hashes=blob_hashes, access_list=access_list, y_parity=y_parity, r=r, s=s,
            )
            return CancunTypedTransaction(3, inner_tx)
        else:
            return CancunLegacyTransaction(
                nonce=nonce, gas_price=gas_price, gas=gas, to=to_addr.canonical_address if to_addr else b'',
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
        )

    def fork_at_block(self, block_number: int):
        block_id = to_hex(block_number)
        self.env.fork_rpc(self.rpc, block_identifier=block_id)

    def _fetch_recent_block_hashes(self, current_block: int):
        w3 = web3.Web3(web3.Web3.HTTPProvider(self.rpc_url))
        logger.info(f"Fetching recent block hashes for block {current_block}")
        start_block = max(0, current_block - 255)
        block_hashes = []
        if current_block > 0:
            block_hashes.append(self._safe_to_bytes(self.context.parent_hash))
        for block_num in range(current_block - 2, start_block - 1, -1):
            if block_num < 0: break
            try:
                block = w3.eth.get_block(block_num)
                block_hashes.append(self._safe_to_bytes(block['hash']))
            except Exception as e:
                logger.warning(f"Failed to fetch block hash for block {block_num}: {e}")
                block_hashes.append(b'\x00' * 32)
        self.recent_block_hashes = block_hashes
        logger.info(f"Fetched {len(self.recent_block_hashes)} block hashes")

    def override_execution_context(self):
        self.env.evm.vm.state.execution_context._base_fee_per_gas = self.context.block_base_fee
        self.env.evm.vm.state.execution_context._coinbase = self._safe_to_bytes(self.context.coinbase)
        self.env.evm.vm.state.execution_context._timestamp = self._safe_to_int(self.context.block_timestamp)
        self.env.evm.vm.state.execution_context._block_number = self._safe_to_int(self.context.block_number)
        self.env.evm.vm.state.execution_context._gas_limit = self._safe_to_int(self.context.block_gas_limit)
        self.env.evm.vm.state.execution_context._mix_hash = self._safe_to_bytes(self.context.mix_hash)
        
        if self.context.excess_blob_gas is not None:
            self.env.evm.vm.state.execution_context._excess_blob_gas = self._safe_to_int(self.context.excess_blob_gas)
        
        # Set recent block hashes for BLOCKHASH opcode support
        self.env.evm.vm.state.execution_context._prev_hashes = self.recent_block_hashes
    

    def _convert_result_to_simulated_order(self, order: Order, result: OrderSimResult) -> SimulatedOrder:
        if result.success:
            sim_value = SimValue(
                coinbase_profit=result.coinbase_profit, gas_used=result.gas_used,
                blob_gas_used=result.blob_gas_used, paid_kickbacks=result.paid_kickbacks
            )
            return SimulatedOrder(order=order, sim_value=sim_value, used_state_trace=result.state_trace)
        else:
            sim_value = SimValue(coinbase_profit=0, gas_used=0, blob_gas_used=0, paid_kickbacks=0)
            simulated_order = SimulatedOrder(order=order, sim_value=sim_value, used_state_trace=result.state_trace)
            simulated_order._error_result = result
            return simulated_order

    def _calculate_coinbase_profit(self, initial_balance: int, final_balance: int) -> int:
        return max(0, final_balance - initial_balance)

    def _decode_revert_output(self, output_hex: str) -> str:
        return self.decode_revert_reason(output_hex)

    def decode_revert_reason(self, revert_data: str) -> str:
        if not revert_data: return "No revert reason provided"
        if revert_data.startswith('0x'): revert_data = revert_data[2:]
        try:
            data = bytes.fromhex(revert_data)
            if len(data) >= 4 and data[:4] == bytes.fromhex('08c379a0'):
                return decode(['string'], data[4:])[0]
            else:
                try: return data.decode('utf-8').strip('\x00')
                except: return f"Custom error or non-string revert: {revert_data}"
        except Exception as e:
            return f"Failed to decode revert reason: {e}"

    def _safe_to_int(self, value: int | str | bytes) -> int:
        if not value: return 0
        if isinstance(value, int): return value
        if isinstance(value, bytes): return int.from_bytes(value, byteorder='big')
        if value == "0x": return 0
        return int(value, 16)

    def _safe_to_bytes(self, value) -> bytes:
        if value is None: return b''
        elif isinstance(value, bytes): return value
        elif isinstance(value, str):
            if value.startswith("b'") and value.endswith("'"):
                try: return eval(value)
                except: pass
            return bytes.fromhex(value.removeprefix("0x"))
        else:
            return bytes(value)

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

        # 1. Get initial on-chain nonces once.
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
        logger.info(f"Block {simulator.context.block_number} simulation completed: {successful} successful, {failed} failed")

        return sim_results_final

    except Exception as e:
        logger.error(f"Block simulation failed: {e}", exc_info=True)
        raise
