import logging
from backtest.common.block_data import BlockData
from backtest.common.order import Order, OrderType, TxOrder, BundleOrder, ShareBundleOrder
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum


from pyrevm import EVM, Env, BlockEnv

logger = logging.getLogger(__name__)


@dataclass
class NonceKey:
    """Key for tracking nonce dependencies"""
    address: str
    nonce: int

    def __hash__(self):
        return hash((self.address, self.nonce))

    def __eq__(self, other):
        return isinstance(other, NonceKey) and self.address == other.address and self.nonce == other.nonce


@dataclass
class SimulationContext:
    block_number: int
    block_timestamp: int  # Unix timestamp
    block_base_fee: int  # Base fee in wei
    block_gas_limit: int  # Block gas limit
    block_hash: Optional[str] = None
    parent_hash: Optional[str] = None
    chain_id: int = 1
    coinbase: str = "0x0000000000000000000000000000000000000000"  # fee recipient address

    # Additional fields for proper block header construction
    block_difficulty: int = 0  # PoS era, always 0
    block_gas_used: int = 0
    withdrawals_root: Optional[str] = None
    blob_gas_used: Optional[int] = None
    excess_blob_gas: Optional[int] = None

    @classmethod
    def from_onchain_block(cls, onchain_block: dict, winning_bid_trace: dict = None) -> 'SimulationContext':
        """
        Create SimulationContext from onchain block data.
        This mirrors rbuilder's BlockBuildingContext::from_onchain_block()
        """

        # Extract all the fields we need for proper block header construction
        context = cls(
            block_number=onchain_block.get('number', 0),
            block_timestamp=onchain_block.get('timestamp', 0),
            block_base_fee=onchain_block.get('baseFeePerGas', 0),
            block_gas_limit=onchain_block.get('gasLimit', 36000000),
            block_hash=onchain_block.get('hash'),
            parent_hash=onchain_block.get('parentHash'),
            chain_id=1,  # Ethereum mainnet
            coinbase=onchain_block.get('miner', "0x0000000000000000000000000000000000000000"),
            block_difficulty=onchain_block.get('difficulty', 0),
            block_gas_used=onchain_block.get('gasUsed', 0),
            withdrawals_root=onchain_block.get('withdrawalsRoot'),
            blob_gas_used=onchain_block.get('blobGasUsed'),
            excess_blob_gas=onchain_block.get('excessBlobGas')
        )

        # If we have winning bid trace, use the fee recipient from there
        # This matches rbuilder's logic of using suggested_fee_recipient
        if winning_bid_trace and 'proposer_fee_recipient' in winning_bid_trace:
            context.coinbase = winning_bid_trace['proposer_fee_recipient']

        return context

class SimulationError(Enum):
    """Types of simulation errors that prevent transaction execution"""
    INSUFFICIENT_BALANCE = "insufficient_balance"
    INVALID_NONCE = "invalid_nonce"
    GAS_LIMIT_EXCEEDED = "gas_limit_exceeded"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class OrderSimResult:
    """Internal result of simulating an order - minimal structure for pybuilder"""
    success: bool
    gas_used: int
    coinbase_profit: int = 0  # in wei
    blob_gas_used: int = 0
    paid_kickbacks: int = 0  # simplified - just total value in wei
    error: Optional[SimulationError] = None
    error_message: Optional[str] = None
    state_changes: Optional[Dict[str, Any]] = None


@dataclass
class SimValue:
    """Economic value of a simulation, matching rbuilder's SimValue"""
    coinbase_profit: int  # in wei
    gas_used: int
    blob_gas_used: int
    paid_kickbacks: int  # in wei


@dataclass
class SimulatedOrder:
    """An order with its simulation results, matching rbuilder's SimulatedOrder"""
    order: Order
    sim_value: SimValue
    used_state_trace: Optional[Any] = None  # Deferred for now
    _error_result: Optional[OrderSimResult] = None  # For failed orders

    @property
    def simulation_result(self) -> OrderSimResult:
        """Backwards compatibility property"""
        if self._error_result is not None:
            # Return the error result for failed orders
            return self._error_result
        else:
            # Return success result for successful orders
            return OrderSimResult(
                success=True,
                gas_used=self.sim_value.gas_used,
                coinbase_profit=self.sim_value.coinbase_profit,
                blob_gas_used=self.sim_value.blob_gas_used,
                paid_kickbacks=self.sim_value.paid_kickbacks
            )

class EVMSimulator:
    def __init__(self, simulation_context: SimulationContext, rpc_url: str):
        self.context = simulation_context
        
        block_env = BlockEnv(
            number=simulation_context.block_number,
            coinbase=simulation_context.coinbase,
            timestamp=simulation_context.block_timestamp,
            difficulty=simulation_context.block_difficulty,
            prevrandao=None,
            basefee=simulation_context.block_base_fee,
            gas_limit=simulation_context.block_gas_limit,
            excess_blob_gas=simulation_context.excess_blob_gas or 0,
        )
        pyenv = Env(block=block_env)
        
        # Use regular EVM without tracing
        self.evm = EVM(
            fork_url=rpc_url,
            fork_block=str(simulation_context.block_number - 1),
            env=pyenv,
        )
            
        # Track nonces and simulated orders
        self.nonce_cache: Dict[str, int] = {}
        self.simulated_orders: Dict[NonceKey, Order] = {}

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

    def _safe_to_int(self, value) -> int:
        """Convert a value to int, handling both hex strings and integers."""
        if isinstance(value, int):
            return value
        elif isinstance(value, str):
            if value == "0x":
                return 0
            return int(value, 16)
        else:
            return int(value)
  
    def _safe_to_bytes(self, value) -> bytes:
        """Convert a value to bytes, handling both hex strings and bytes."""
        if isinstance(value, bytes):
            return value
        elif isinstance(value, str):
            return bytes.fromhex(value.removeprefix("0x"))
        else:
            return bytes(value)

    def _execute_tx(self, tx_data) -> OrderSimResult:
        """Simulate a transaction execution."""

        # Extract transaction parameters
        tx_type = tx_data["type"]
        tx_type_int = self._safe_to_int(tx_type) if tx_type else 0
        from_addr = tx_data["from"]
        to_addr = tx_data["to"] if tx_data["to"] else None
        value = self._safe_to_int(tx_data["value"])
        gas = self._safe_to_int(tx_data["gas"])
        gas_price = self._safe_to_int(tx_data["gasPrice"])
        data = self._safe_to_bytes(tx_data["input"])


        if tx_type_int == 2:  # EIP-1559 transaction
            max_fee_per_gas = self._safe_to_int(tx_data["maxFeePerGas"])
            max_priority_fee_per_gas = self._safe_to_int(tx_data["maxPriorityFeePerGas"])
            gas_price = min(max_fee_per_gas, self.context.block_base_fee + max_priority_fee_per_gas)
        else:
            gas_price = self._safe_to_int(tx_data["gasPrice"])
            max_priority_fee_per_gas = None

        # Snapshot state and measure coinbase
        checkpoint = self.evm.snapshot()
        cb = self.context.coinbase
        before_cb = self.evm.get_balance(cb)

        # Build message_call args
        msg_args: Dict[str, Any] = {
            "caller": from_addr,
            "to": to_addr,
            "value": value,
            "calldata": data,
            "gas_price": gas_price,
            "gas": gas,
        }

        try:

            self.evm.message_call(**msg_args)
            
            profit = self.evm.get_balance(cb) - before_cb - self.context.block_base_fee * self.evm.result.gas_used
            self.evm.set_balance(cb, before_cb + profit)
            self.evm.commit()

            return OrderSimResult(
                success=True,
                gas_used=self.evm.result.gas_used,
                coinbase_profit=profit,
                blob_gas_used=getattr(self.evm.result, 'blob_gas_used', 0),
                paid_kickbacks=0,
            )

        except Exception as e:

            # Check if the transaction used gas before failing
            gas_used = getattr(self.evm.result, 'gas_used', 0) if hasattr(self.evm, 'result') else 0
            
            if gas_used > 0:
                # Transaction failed after using gas - builder can still collect fees
                profit = self.evm.get_balance(cb) - before_cb - self.context.block_base_fee * gas_used
                self.evm.set_balance(cb, before_cb + profit)
                self.evm.commit()
                
                logger.info(f"Transaction reverted but used gas ({gas_used}), treating as successful: {e}")
                return OrderSimResult(
                    success=True,
                    gas_used=gas_used,
                    coinbase_profit=profit,
                    blob_gas_used=getattr(self.evm.result, 'blob_gas_used', 0) if hasattr(self.evm, 'result') else 0,
                    paid_kickbacks=0,
                )
            else:
                # Transaction failed before using gas - validation error
                logger.error(f"Transaction simulation failed during validation: {e}")
                return OrderSimResult(
                    success=False,
                    gas_used=0,
                    coinbase_profit=0,
                    blob_gas_used=0,
                    paid_kickbacks=0,
                    error=SimulationError.VALIDATION_ERROR,
                    error_message=str(e),
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
        rollback = self.evm.snapshot()
        total_gas, total_profit = 0, 0
        for child_order, optional in order.child_orders:
            res = self._execute_tx(child_order.get_transaction_data())
            if not res.success and not optional:
                self.evm.revert(rollback)
                return self._convert_result_to_simulated_order(order, res)
            if res.success:
                total_gas += res.gas_used
                total_profit += res.coinbase_profit
        self.evm.revert(rollback)
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

        rollback = self.evm.snapshot()
        try:
            # First simulate all parent orders, updating nonce cache
            for parent_order in parent_orders:
                parent_res = self._simulate_single_order(parent_order)
                if not parent_res.simulation_result.success:
                    # Parent failed: rollback and return failure
                    self.evm.revert(rollback)
                    error_result = OrderSimResult(
                        success=False,
                        gas_used=0,
                        coinbase_profit=0,
                        blob_gas_used=0,
                        paid_kickbacks=0,
                        error=SimulationError.VALIDATION_ERROR,
                        error_message=f"Parent order failed: {parent_res.simulation_result.error_message}"
                    )
                    return self._convert_result_to_simulated_order(order, error_result)
                # Record nonce update for parent
                self._record_order_execution(parent_order)

            # Now simulate main order
            main_res = self._simulate_single_order(order)
            if main_res.simulation_result.success:
                self._record_order_execution(order)
            else:
                # On failure of main, rollback
                self.evm.revert(rollback)
            return main_res
        except Exception as e:
            # On exception, rollback and return error
            self.evm.revert(rollback)
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
    
    def simulate_order(self, order: Order) -> SimulatedOrder:
        """
        Simulate any type of order, automatically handling parent dependencies.
        This is the main entry point that mirrors rbuilder's behavior.
        """
        # Check if order dependencies are satisfied
        is_ready, parent_orders = self._check_order_dependencies(order)
        
        if not is_ready:
            error_result = OrderSimResult(
                success=False,
                gas_used=0,
                error=SimulationError.INVALID_NONCE,
                error_message="Order dependencies not satisfied"
            )
            return self._convert_result_to_simulated_order(order, error_result)
        
        return self.simulate_order_with_parents(order, parent_orders)

    def _get_account_nonce(self, address: str) -> int:
        """Get the current nonce for an account, with caching."""
        if address not in self.nonce_cache:
            # Fetch nonce via pyrevm
            account = self.evm.basic(address)
            nonce = int(account.nonce)
            self.nonce_cache[address] = nonce
        return self.nonce_cache[address]
    
    def _update_account_nonce(self, address: str, new_nonce: int):
        """Update the cached nonce for an account and override in EVM state."""
        # Update local cache
        self.nonce_cache[address] = new_nonce
    
    def _check_order_dependencies(self, order: Order) -> Tuple[bool, List[Order]]:
        """
        Check if an order's nonce dependencies are satisfied.
        Returns (is_ready, parent_orders_needed)
        """
        parent_orders = []
        
        for tx_nonce in order.nonces():
            current_nonce = self._get_account_nonce(tx_nonce.address)
            
            if tx_nonce.nonce == current_nonce:
                # This nonce is ready to execute
                continue
            elif tx_nonce.nonce < current_nonce:
                # Nonce is already used
                if not tx_nonce.optional:
                    # Required nonce is invalid, order cannot execute
                    return False, []
                else:
                    # Optional nonce, can skip
                    continue
            else:
                # tx_nonce.nonce > current_nonce, need parent orders
                nonce_key = NonceKey(tx_nonce.address, tx_nonce.nonce)
                if nonce_key in self.simulated_orders:
                    # We have a simulated order that satisfies this nonce
                    parent_orders.append(self.simulated_orders[nonce_key])
                else:
                    # Missing dependency, order is not ready
                    return False, []
        
        return True, parent_orders
    
    def _record_order_execution(self, order: Order):
        """Record that an order has been executed and update nonce tracking"""
        for tx_nonce in order.nonces():
            current_nonce = self._get_account_nonce(tx_nonce.address)
            if tx_nonce.nonce >= current_nonce:
                # Update nonce and record the order
                self._update_account_nonce(tx_nonce.address, tx_nonce.nonce + 1)
                nonce_key = NonceKey(tx_nonce.address, tx_nonce.nonce)
                self.simulated_orders[nonce_key] = order

    def _create_rollback_point(self):
        """Create a rollback point to restore state if needed"""
        # Use the env's anchor context manager functionality
        return self.env.anchor()
    
    def _rollback_to_point(self, rollback_point):
        """Rollback to a previous state point"""
        # The anchor context manager handles rollback automatically when exited
        # This is a placeholder (the actual rollback happens in the context manager)
        pass

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
                used_state_trace=None  # TODO: Add state tracing
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
                used_state_trace=None
            )
            # Store error info for backwards compatibility
            simulated_order._error_result = result
            return simulated_order

    def _calculate_coinbase_profit(self, initial_balance: int, final_balance: int) -> int:
        """Calculate coinbase profit from balance change"""
        return max(0, final_balance - initial_balance)

def simulate_orders(orders: List[Order], block_data: BlockData, rpc_url: str) -> List[SimulatedOrder]:
    """
    Simulate orders using onchain block data for proper context.
    Implements order dependency resolution similar to rbuilder's SimTree.
    
    Args:
        orders: List of orders to simulate
        block_data: Block data containing onchain block and winning bid trace
        rpc_url: RPC URL for EVM simulation
        
    Returns:
        List of simulated orders with results
    """
    try:
        context = SimulationContext.from_onchain_block(block_data.onchain_block)
        logger.info(f"Simulating {len(orders)} orders for block {context.block_number}")
        logger.info(f"Using onchain block data: hash={context.block_hash}")

        simulator = EVMSimulator(context, rpc_url)
        logger.info(f"Using EVM-based simulation with onchain block context (parent state root)")
        
        results = []
        
        # Sort orders to process dependencies properly
        # Simple approach: process orders in multiple passes until no more can be processed
        remaining_orders = orders.copy()
        max_passes = len(orders) + 1  # Safety limit to prevent infinite loops
        
        for pass_num in range(max_passes):
            if not remaining_orders:
                break
                
            processed_this_pass = False
            
            for order in remaining_orders.copy():
                # Check if this order's dependencies are satisfied
                is_ready, parent_orders = simulator._check_order_dependencies(order)
                
                if is_ready:
                    logger.debug(f"Pass {pass_num}: Simulating order {order.id()} with {len(parent_orders)} parents")
                    result = simulator.simulate_order_with_parents(order, parent_orders)
                    results.append(result)
                    processed_this_pass = True

                    # Update remaining orders list
                    remaining_orders.remove(order)
                    
            # If we didn't process any orders in this pass, remaining orders have unresolvable dependencies
            if not processed_this_pass:
                # Add failed orders to results
                for order in remaining_orders:
                    error_result = OrderSimResult(
                        success=False,
                        gas_used=0,
                        error=SimulationError.INVALID_NONCE,
                        error_message="Unresolvable nonce dependencies"
                    )
                    failed_result = SimulatedOrder(
                        order=order,
                        sim_value=SimValue(
                            coinbase_profit=0,
                            gas_used=0,
                            blob_gas_used=0,
                            paid_kickbacks=0
                        )
                    )
                    # Add error info to the failed result
                    failed_result._error_result = error_result
                    results.append(failed_result)
                break
        
        # Log summary
        successful = sum(1 for r in results if r.simulation_result.success)
        failed = len(results) - successful
        total_gas = sum(r.simulation_result.gas_used for r in results if r.simulation_result.success)
        
        logger.info(f"Block {context.block_number} simulation completed: {successful} successful, {failed} failed")
        logger.info(f"Total gas used: {total_gas}")

        return results
        
    except Exception as e:
        logger.error(f"Block simulation failed: {e}")
        raise
