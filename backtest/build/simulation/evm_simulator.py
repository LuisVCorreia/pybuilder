import logging
from backtest.common.block_data import BlockData
from backtest.common.order import Order, OrderType, TxOrder, BundleOrder, ShareBundleOrder
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

from boa.vm.py_evm import Address
from boa.rpc import EthereumRPC, to_hex, to_int, to_bytes
from boa.environment import Env

logger = logging.getLogger(__name__)


@dataclass
class AccountInfo:
    """Ethereum account information"""
    balance: int  # in wei
    nonce: int
    bytecode_hash: str  # 0x-prefixed hex


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
    block_gas_used: int = 0  # Starting gas used
    withdrawals_root: Optional[str] = None  # For post-Shanghai blocks
    blob_gas_used: Optional[int] = None  # For post-Cancun blocks 
    excess_blob_gas: Optional[int] = None  # For post-Cancun blocks
    
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
    def __init__(self,simulation_context: SimulationContext, rpc_url: str):
        self.context = simulation_context
        self.rpc = EthereumRPC(rpc_url)
        self.env = Env(fast_mode_enabled=True, fork_try_prefetch_state=True)    # Creates new environment with Py-EVM execution and remote RPC support


    def fork_at_block(self, block_number: int):
        """Fork the EVM state at the specified block number."""
        block_id = to_hex(block_number)        
        # Use the env's fork_rpc method instead of direct EVM access
        self.env.fork_rpc(self.rpc, block_identifier=block_id)
        
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
                    gas_refund = computation.get_gas_refund()
                    total_gas_used = intrinsic_gas + execution_gas_used - gas_refund
                    
                    # Collect results
                    result = SimulationResult(
                        success=computation.is_success,
                        gas_used=total_gas_used,
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
        logger.info(f"Simulating transaction {order.id()}")

        # Only accept TxOrder
        if not isinstance(order, TxOrder):
            logger.error(f"simulate_tx_order called with non-TxOrder: {type(order)}")
            return SimulatedOrder(
                order=order,
                simulation_result=SimulationResult(
                    success=False,
                    gas_used=0,
                    error=SimulationError.UNKNOWN_ERROR,
                    error_message="simulate_tx_order called with non-TxOrder"
                )
            )
        try:
            self.fork_at_block(self.context.block_number - 1)

            tx_data = order.get_transaction_data()
            result = self._execute_tx(tx_data)
            logger.info(f"Simulation result: {result.success}, gas used: {result.gas_used}")
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
                    error=SimulationError.UNKNOWN_ERROR,
                    error_message="simulate_bundle_order called with non-BundleOrder"
                )
            )
        try:
            bundle_rollback_point = self._create_rollback_point()
            total_gas_used = 0
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
                else:
                    logger.warning(f"Bundle contains non-transaction order type: {getattr(child_order, 'order_type', lambda: '?')()}")
            if not bundle_success:
                self._rollback_to_point(bundle_rollback_point)
                total_gas_used = 0
            result = SimulationResult(
                success=bundle_success,
                gas_used=total_gas_used,
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
                    error=SimulationError.UNKNOWN_ERROR,
                    error_message="Order does not have order_type method."
                )
            )

def simulate_orders(orders: List[Order], block_data: BlockData, rpc_url: str) -> List[SimulatedOrder]:
    """
    Simulate orders using onchain block data for proper context.
    
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
        for order in orders:
            result = simulator.simulate_order(order)
            results.append(result)

        # Log summary
        successful = sum(1 for r in results if r.simulation_result.success)
        failed = len(results) - successful
        total_gas = sum(r.simulation_result.gas_used for r in results if r.simulation_result.success)
        total_profit = sum(r.simulation_result.coinbase_profit for r in results if r.simulation_result.success)
        
        logger.info(f"Block {context.block_number} simulation completed: {successful} successful, {failed} failed")
        logger.info(f"Total gas used: {total_gas}, Total coinbase profit: {total_profit} wei")
        
        return results
        
    except Exception as e:
        logger.error(f"Block simulation failed: {e}")
        raise
