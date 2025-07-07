import logging
from backtest.common.order import Order, OrderType, TxOrder, BundleOrder, ShareBundleOrder
from .state_provider import StateProvider, SimulationContext
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

from boa.vm.py_evm import Address
from boa.rpc import EthereumRPC, to_hex, to_int, to_bytes
from boa.environment import Env

logger = logging.getLogger(__name__)

class SimulationError(Enum):
    """Types of simulation errors"""
    INSUFFICIENT_BALANCE = "insufficient_balance"
    INVALID_NONCE = "invalid_nonce"
    GAS_LIMIT_EXCEEDED = "gas_limit_exceeded"
    EXECUTION_REVERTED = "execution_reverted"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class SimulationResult:
    """Result of simulating an order"""
    success: bool
    gas_used: int
    coinbase_profit: int  # in wei
    error: Optional[SimulationError] = None
    error_message: Optional[str] = None
    state_changes: Optional[Dict[str, Any]] = None


@dataclass
class SimulatedOrder:
    """An order with its simulation results"""
    order: Order
    simulation_result: SimulationResult
    
    @property
    def sim_value(self):
        """Alias for compatibility with rbuilder patterns"""
        return self.simulation_result

class EVMSimulator:
    def __init__(self,
                 state_provider: StateProvider,
                 simulation_context: SimulationContext):
        self.state_provider = state_provider
        self.context = simulation_context

        self.rpc = EthereumRPC(state_provider.rpc_url)
        self.env = Env(fast_mode_enabled=True, fork_try_prefetch_state=True)    # Creates new environment with Py-EVM execution and remote RPC support


    def fork_at_block(self, block_number: int):
        """Fork the EVM state at the specified block number."""
        block_id = to_hex(block_number)        
        # Use the env's fork_rpc method instead of direct EVM access
        self.env.fork_rpc(self.rpc, block_identifier=block_id)
        
        print(f"Successfully forked at block {block_number}")

    def _safe_to_int(self, value) -> int:
        """Convert a value to int, handling both hex strings and integers."""
        if isinstance(value, int):
            return value
        elif isinstance(value, str):
            return to_int(value)
        else:
            return int(value)
    
    def _safe_to_bytes(self, value) -> bytes:
        """Convert a value to bytes, handling both hex strings and bytes."""
        if isinstance(value, bytes):
            return value
        elif isinstance(value, str):
            return to_bytes(value)
        else:
            return bytes(value)

    def _execute_tx(self, tx_data) -> SimulationResult:
        """Simulate a transaction execution."""

        # Fork the EVM state at the current block
        self.fork_at_block(self.context.block_number)

        # Extract transaction parameters
        from_addr = Address(tx_data["from"])
        to_addr = Address(tx_data["to"]) if tx_data["to"] else None
        value = self._safe_to_int(tx_data["value"])
        gas = self._safe_to_int(tx_data["gas"])
        gas_price = self._safe_to_int(tx_data["gasPrice"])
        data = self._safe_to_bytes(tx_data["input"])

        # Get initial balances for validation
        initial_from_balance = self.env.get_balance(from_addr)
        initial_to_balance = self.env.get_balance(to_addr) if to_addr else 0

        with self.env.anchor():
            
            try:
                # Execute the transaction
                print("Executing transaction...")
                
                # Get the code at the target address if it's a contract call
                code = b""
                if to_addr:
                    code = self.env.get_code(to_addr)
                
                # Calculate intrinsic gas cost (base transaction cost)
                intrinsic_gas = 21000  # Base cost for any transaction
                
                # Add gas for calldata (input data)
                if len(data) > 0:
                    for byte in data:
                        if byte == 0:
                            intrinsic_gas += 4  # Cost for zero byte
                        else:
                            intrinsic_gas += 16  # Cost for non-zero byte
                            
                # For simple ETH transfers, we need to handle this differently
                is_simple_transfer = len(data) == 0 and (not to_addr or len(code) == 0)
                
                if is_simple_transfer:
                    
                    # Check if sender has enough balance
                    if initial_from_balance < value:
                        result = SimulationResult(
                            success=False,
                            gas_used=intrinsic_gas,
                            coinbase_profit=0,
                            error=SimulationError.INSUFFICIENT_BALANCE,
                            error_message="Insufficient balance for transfer"
                        )
                    else:
                        # Perform the transfer
                        self.env.set_balance(from_addr, initial_from_balance - value)
                        if to_addr:
                            self.env.set_balance(to_addr, initial_to_balance + value)

                        result = SimulationResult(
                            success=True,
                            gas_used=intrinsic_gas,
                            coinbase_profit=0,  # TODO: handle coinbase profit
                            error=None,
                            error_message=None,
                            state_changes={
                                "balances": {
                                    from_addr: initial_from_balance - value,
                                    to_addr: initial_to_balance + value
                                }
                            }
                        )

                else:
                    
                    # Execute the transaction using the EVM for contract calls
                    computation = self.env.execute_code(
                        to_address=to_addr,
                        sender=from_addr,
                        gas=gas - intrinsic_gas,  # Subtract intrinsic gas from available gas
                        value=value,
                        override_bytecode=code,
                        data=data,
                        is_modifying=True,
                        simulate=False,
                    )
                    
                    # Calculate total gas used (intrinsic + execution)
                    execution_gas_used = computation.get_gas_used()
                    total_gas_used = intrinsic_gas + execution_gas_used
                    
                    # Collect results
                    result = SimulationResult(
                        success=computation.is_success,
                        gas_used=total_gas_used,
                        coinbase_profit=0,  # TODO: handle coinbase profit
                        error=None if computation.is_success else SimulationError.EXECUTION_REVERTED,
                        error_message=None if computation.is_success else "Transaction execution reverted",
                        state_changes={
                            "balances": {
                                from_addr: initial_from_balance - value,
                                to_addr: initial_to_balance + value if to_addr else initial_to_balance
                            },
                            "logs": computation.get_log_entries()
                        }
                    )

                return result
                
            except Exception as e:
                raise
                # The anchor context manager will automatically revert the state


    def simulate_tx_order(self, order: TxOrder) -> SimulatedOrder:
        # Only accept TxOrder
        if not isinstance(order, TxOrder):
            logger.error(f"simulate_tx_order called with non-TxOrder: {type(order)}")
            return SimulatedOrder(
                order=order,
                simulation_result=SimulationResult(
                    success=False,
                    gas_used=0,
                    coinbase_profit=0,
                    error=SimulationError.UNKNOWN_ERROR,
                    error_message="simulate_tx_order called with non-TxOrder"
                )
            )
        try:
            # Only call get_transaction_data if present
            tx_data = order.get_transaction_data() if hasattr(order, 'get_transaction_data') else None
            if not tx_data:
                tx_data = {
                    'from': getattr(order, 'sender', '0x' + '0' * 40),
                    'nonce': getattr(order, 'nonce', 0),
                    'gasPrice': getattr(order, 'max_fee_per_gas', 0),
                    'gas': 21000,
                    'to': None,
                    'value': 0,
                    'input': '0x'
                }
            result = self._execute_tx(tx_data)
            return SimulatedOrder(
                order=order,
                simulation_result=result
            )
        except Exception as e:
            logger.error(f"Failed to simulate tx order {getattr(order, 'id', lambda: '?')()}: {e}")
            return SimulatedOrder(
                order=order,
                simulation_result=SimulationResult(
                    success=False,
                    gas_used=0,
                    coinbase_profit=0,
                    error=SimulationError.EXECUTION_REVERTED,
                    error_message=str(e)
                )
            )
    
    def simulate_bundle_order(self, order: BundleOrder) -> SimulatedOrder:
        # Only accept BundleOrder
        if not isinstance(order, BundleOrder):
            logger.error(f"simulate_bundle_order called with non-BundleOrder: {type(order)}")
            return SimulatedOrder(
                order=order,
                simulation_result=SimulationResult(
                    success=False,
                    gas_used=0,
                    coinbase_profit=0,
                    error=SimulationError.UNKNOWN_ERROR,
                    error_message="simulate_bundle_order called with non-BundleOrder"
                )
            )
        try:
            bundle_rollback_point = self._create_rollback_point()
            total_gas_used = 0
            total_coinbase_profit = 0
            bundle_success = True
            error_message = None
            for i, (child_order, optional) in enumerate(order.child_orders):
                if hasattr(child_order, 'order_type') and child_order.order_type() == OrderType.TX:
                    tx_data = child_order.get_transaction_data() if hasattr(child_order, 'get_transaction_data') else None
                    if not tx_data:
                        tx_data = {
                            'from': getattr(child_order, 'sender', '0x' + '0' * 40),
                            'nonce': getattr(child_order, 'nonce', 0),
                            'gasPrice': getattr(child_order, 'max_fee_per_gas', 0),
                            'gas': 21000,
                            'to': None,
                            'value': 0,
                            'input': '0x'
                        }
                    tx_result = self._execute_tx(tx_data)
                    if not tx_result.success and not optional:
                        bundle_success = False
                        error_message = f"Required transaction {i} failed: {tx_result.error_message}"
                        break
                    elif tx_result.success:
                        total_gas_used += tx_result.gas_used
                        total_coinbase_profit += tx_result.coinbase_profit
                else:
                    logger.warning(f"Bundle contains non-transaction order type: {getattr(child_order, 'order_type', lambda: '?')()}")
            if not bundle_success:
                self._rollback_to_point(bundle_rollback_point)
                total_gas_used = 0
                total_coinbase_profit = 0
            result = SimulationResult(
                success=bundle_success,
                gas_used=total_gas_used,
                coinbase_profit=total_coinbase_profit,
                error=None if bundle_success else SimulationError.EXECUTION_REVERTED,
                error_message=error_message
            )
            return SimulatedOrder(
                order=order,
                simulation_result=result
            )
        except Exception as e:
            self._rollback_to_point(bundle_rollback_point)
            logger.error(f"Failed to simulate bundle order {getattr(order, 'id', lambda: '?')()}: {e}")
            return SimulatedOrder(
                order=order,
                simulation_result=SimulationResult(
                    success=False,
                    gas_used=0,
                    coinbase_profit=0,
                    error=SimulationError.EXECUTION_REVERTED,
                    error_message=str(e)
                )
            )
    
    def simulate_share_bundle_order(self, order: ShareBundleOrder) -> SimulatedOrder:
        # Only accept ShareBundleOrder
        if not isinstance(order, ShareBundleOrder):
            logger.error(f"simulate_share_bundle_order called with non-ShareBundleOrder: {type(order)}")
            return SimulatedOrder(
                order=order,
                simulation_result=SimulationResult(
                    success=False,
                    gas_used=0,
                    coinbase_profit=0,
                    error=SimulationError.UNKNOWN_ERROR,
                    error_message="simulate_share_bundle_order called with non-ShareBundleOrder"
                )
            )
        return self.simulate_bundle_order(order)  # Only call with correct type
    
    def simulate_order(self, order: Order) -> SimulatedOrder:
        """Simulate any type of order"""
        # Only call the correct simulation method for the order type
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
                    return SimulatedOrder(
                        order=order,
                        simulation_result=SimulationResult(
                            success=False,
                            gas_used=0,
                            coinbase_profit=0,
                            error=SimulationError.UNKNOWN_ERROR,
                            error_message="ShareBundleOrder simulation not supported"
                        )
                    )
            else:
                logger.error(f"Unknown or unsupported order type: {otype}")
                return SimulatedOrder(
                    order=order,
                    simulation_result=SimulationResult(
                        success=False,
                        gas_used=0,
                        coinbase_profit=0,
                        error=SimulationError.UNKNOWN_ERROR,
                        error_message=f"Order type {otype} not supported"
                    )
                )
        else:
            logger.error("Order does not have order_type method.")
            return SimulatedOrder(
                order=order,
                simulation_result=SimulationResult(
                    success=False,
                    gas_used=0,
                    coinbase_profit=0,
                    error=SimulationError.UNKNOWN_ERROR,
                    error_message="Order does not have order_type method."
                )
            )
