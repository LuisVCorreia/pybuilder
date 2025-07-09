import logging
from backtest.common.block_data import BlockData
from backtest.common.order import Order, OrderType, TxOrder, BundleOrder, ShareBundleOrder, TxNonce
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from boa.vm.py_evm import Address
from boa.rpc import EthereumRPC, to_hex, to_int, to_bytes
from boa.environment import Env

from eth.vm.forks.london.transactions import DynamicFeeTransaction
from eth.vm.forks.cancun.transactions import CancunTypedTransaction
from eth.vm.forks.cancun.transactions import CancunLegacyTransaction
from eth.vm.forks.cancun.headers import CancunBlockHeader

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
    def __init__(self,simulation_context: SimulationContext, rpc_url: str):
        self.context = simulation_context
        self.rpc = EthereumRPC(rpc_url)
        self.env = Env(fast_mode_enabled=True, fork_try_prefetch_state=False)    # Creates new environment with Py-EVM execution and remote RPC support
        # Track nonces for dependency resolution
        self.nonce_cache: Dict[str, int] = {}
        self.simulated_orders: Dict[NonceKey, Order] = {}  # Track which orders have been simulated for each nonce

        self.fork_at_block(self.context.block_number - 1)  # Fork at the parent block to simulate correctly

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

    def _execute_tx(self, tx_data) -> OrderSimResult:
        """Simulate a transaction execution."""

        # Extract transaction parameters
        tx_type = tx_data["type"]
        tx_type_int = self._safe_to_int(tx_type) if tx_type else 0
        from_addr = Address(tx_data["from"])
        to_addr = Address(tx_data["to"]) if tx_data["to"] else None
        value = self._safe_to_int(tx_data["value"])
        gas = self._safe_to_int(tx_data["gas"])
        gas_price = self._safe_to_int(tx_data["gasPrice"])
        data = self._safe_to_bytes(tx_data["input"])
        nonce = self._safe_to_int(tx_data["nonce"])

        # Extract signature fields
        r = self._safe_to_int(tx_data["r"])
        s = self._safe_to_int(tx_data["s"])


        if tx_type_int == 2:  # EIP-1559 transaction
            y_parity = self._safe_to_int(tx_data["y_parity"])
            max_fee_per_gas = self._safe_to_int(tx_data["maxFeePerGas"])
            max_priority_fee_per_gas = self._safe_to_int(tx_data["maxPriorityFeePerGas"])
        else:
            v = self._safe_to_int(tx_data["v"])
            gas_price = self._safe_to_int(tx_data["gasPrice"])

        # Get initial balances
        initial_from_balance = self.env.get_balance(from_addr)
        initial_to_balance = self.env.get_balance(to_addr) if to_addr else 0

        coinbase_addr = self.env.evm.vm.state.coinbase  # TODO: Find way to context.coinbase
        initial_coinbase_balance = self.env.get_balance(coinbase_addr)

        # Calculate intrinsic gas cost (base transaction cost)
        intrinsic_gas = 21000  # Base cost for any transaction

        # Add gas for calldata (input data)
        if len(data) > 0:
            for byte in data:
                if byte == 0:
                    intrinsic_gas += 4  # Cost for zero byte
                else:
                    intrinsic_gas += 16  # Cost for non-zero byte

        # Pre-execution validation checks (these should mark transactions as failed)
        # TODO: Since we are now using apply_transaction, we can remove these validation checks
        # Apply_transaction begins by running validation checks on the tx
        # Need to check if error is of type ValidationError
        total_tx_cost = value + (gas * gas_price)
        if initial_from_balance < total_tx_cost:
            return OrderSimResult(
                success=False,
                gas_used=0,
                error=SimulationError.INSUFFICIENT_BALANCE,
                error_message=f"Insufficient balance: need {total_tx_cost}, have {initial_from_balance}"
            )

        if gas < intrinsic_gas:
            return OrderSimResult(
                success=False,
                gas_used=0,
                error=SimulationError.GAS_LIMIT_EXCEEDED,
                error_message=f"Gas limit too low: need {intrinsic_gas}, have {gas}"
            )

        try:
            if tx_type_int == 2:                 
                # Create Type 2 transaction object
                inner_tx = DynamicFeeTransaction(
                    chain_id=1,  # Mainnet
                    nonce=nonce,
                    max_priority_fee_per_gas=max_priority_fee_per_gas,
                    max_fee_per_gas=max_fee_per_gas,
                    gas=gas,
                    to=to_addr.canonical_address if to_addr else b'',
                    value=value,
                    data=data,
                    access_list=[],  # Empty access list for now
                    y_parity=y_parity,
                    r=r,
                    s=s,
                )
                
                tx = CancunTypedTransaction(2, inner_tx)
                
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
            
            vm = self.env.evm.vm
            vm_header = vm.get_header()  # Default block header
            
            # Create proper block header using simulation context
            header = CancunBlockHeader(
                parent_hash=self.context.parent_hash if self.context.parent_hash else vm_header.parent_hash,
                uncles_hash=vm_header.uncles_hash,
                coinbase=to_bytes(self.context.coinbase),
                state_root=vm_header.state_root,
                transaction_root=vm_header.transaction_root,
                receipt_root=vm_header.receipt_root,
                bloom=0,
                difficulty=self.context.block_difficulty,
                block_number=self.context.block_number,
                gas_limit=self.context.block_gas_limit,
                gas_used=0,
                timestamp=self.context.block_timestamp,
                extra_data=b'',
                mix_hash=b'\x00' * 32,
                nonce=b'\x00' * 8,
                base_fee_per_gas=self.context.block_base_fee,
                withdrawals_root=self.context.withdrawals_root,
                blob_gas_used=self.context.blob_gas_used,
                excess_blob_gas=self.context.excess_blob_gas,
            )

            # Apply the transaction with the header
            receipt, computation = vm.apply_transaction(header, tx)

            final_coinbase_balance = self.env.get_balance(coinbase_addr)
            coinbase_profit = self._calculate_coinbase_profit(initial_coinbase_balance, final_coinbase_balance)

            result = OrderSimResult(
                success=True,  # Always true for executed transactions TODO: not anymore now that we're using apply_transaction
                gas_used=receipt.gas_used,
                coinbase_profit=coinbase_profit,
                error=None,
                error_message=None,
                state_changes={
                    "balances": {
                        from_addr: initial_from_balance,
                        to_addr: initial_to_balance,
                        coinbase_addr: initial_coinbase_balance,
                    },
                    "logs": computation.get_log_entries()
                }
            )

            return result

        except Exception as e:
            # Execution errors are treated as validation failures
            print(f"Transaction execution failed: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return OrderSimResult(
                success=False,
                gas_used=0,
                error=SimulationError.VALIDATION_ERROR,
                error_message=f"Transaction execution failed: {str(e)}"
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
        # Only accept BundleOrder
        if not isinstance(order, BundleOrder):
            logger.error(f"simulate_bundle_order called with non-BundleOrder: {type(order)}")
            error_result = OrderSimResult(
                success=False,
                gas_used=0,
                error=SimulationError.UNKNOWN_ERROR,
                error_message="simulate_bundle_order called with non-BundleOrder"
            )
            return self._convert_result_to_simulated_order(order, error_result)
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
            result = OrderSimResult(
                success=bundle_success,
                gas_used=total_gas_used,
                coinbase_profit=total_coinbase_profit,
                error=None if bundle_success else SimulationError.VALIDATION_ERROR,
                error_message=error_message
            )
            return self._convert_result_to_simulated_order(order, result)
        except Exception as e:
            self._rollback_to_point(bundle_rollback_point)
            logger.error(f"Failed to simulate bundle order {getattr(order, 'id', lambda: '?')()}: {e}")
            error_result = OrderSimResult(
                success=False,
                gas_used=0,
                error=SimulationError.VALIDATION_ERROR,
                error_message=str(e)
            )
            return self._convert_result_to_simulated_order(order, error_result)
    
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
        This mirrors rbuilder's simulate_order_using_fork functionality.
        """
        if parent_orders is None:
            parent_orders = []
            
        try:
            # Ensure we're forked at the correct block for the entire parent-child chain
            self.fork_at_block(self.context.block_number - 1)
            
            # Create a rollback point to restore state if needed
            rollback_point = self._create_rollback_point()
            
            # First simulate all parent orders on the same fork
            for parent_order in parent_orders:
                parent_result = self._simulate_single_order(parent_order)
                if not parent_result.simulation_result.success:
                    # Parent failed, rollback and return failure
                    self._rollback_to_point(rollback_point)
                    error_result = OrderSimResult(
                        success=False,
                        gas_used=0,
                        error=SimulationError.VALIDATION_ERROR,
                        error_message=f"Parent order failed: {parent_result.simulation_result.error_message}"
                    )
                    return self._convert_result_to_simulated_order(order, error_result)
                # Record parent execution for nonce tracking (but don't accumulate gas/profit)
                self._record_order_execution(parent_order)
            
            # Now simulate the main order (returns only its own gas/profit)
            result = self._simulate_single_order(order)
            if result.simulation_result.success:
                self._record_order_execution(order)
            else:
                # Main order failed, rollback
                self._rollback_to_point(rollback_point)
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to simulate order with parents {order.id()}: {e}")
            error_result = OrderSimResult(
                success=False,
                gas_used=0,
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
        """Get the current nonce for an account, with caching"""
        if address not in self.nonce_cache:
            # Get nonce from forked state - need canonical address (bytes)
            addr = Address(address)
            nonce = self.env.evm.vm.state.get_nonce(addr.canonical_address)
            self.nonce_cache[address] = nonce
        return self.nonce_cache[address]
    
    def _update_account_nonce(self, address: str, new_nonce: int):
        """Update the cached nonce for an account"""
        self.nonce_cache[address] = new_nonce
        # Also update the EVM state if needed
        addr = Address(address)
        self.env.evm.vm.state.set_nonce(addr.canonical_address, new_nonce)
    
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
