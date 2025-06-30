import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from backtest.common.order import Order, OrderType
from .state_provider import StateProvider, StateProviderFactory, SimulationContext

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


class SimpleOrderSimulator:
    """Simplified order simulator that validates transactions without full EVM execution"""
    
    def __init__(self, state_provider_factory: StateProviderFactory):
        self.state_provider_factory = state_provider_factory
    
    def simulate_order(self, order: Order, context: SimulationContext) -> SimulatedOrder:
        """Simulate a single order"""
        try:
            logger.debug(f"Simulating order {order.id()}")
            
            # Get state provider for the parent block
            state_provider = self.state_provider_factory.history_by_block_number(
                context.block_number - 1
            )
            
            # Simulate based on order type
            if order.order_type() == OrderType.TX:
                result = self._simulate_transaction(order, context, state_provider)
            elif order.order_type() == OrderType.BUNDLE:
                result = self._simulate_bundle(order, context, state_provider)
            elif order.order_type() == OrderType.SHAREBUNDLE:
                result = self._simulate_share_bundle(order, context, state_provider)
            else:
                result = SimulationResult(
                    success=False,
                    gas_used=0,
                    coinbase_profit=0,
                    error=SimulationError.UNKNOWN_ERROR,
                    error_message=f"Unknown order type {order.order_type()}"
                )
            
            return SimulatedOrder(order=order, simulation_result=result)
            
        except Exception as e:
            logger.error(f"Error simulating order {order.id()}: {e}")
            result = SimulationResult(
                success=False,
                gas_used=0,
                coinbase_profit=0,
                error=SimulationError.UNKNOWN_ERROR,
                error_message=str(e)
            )
            return SimulatedOrder(order=order, simulation_result=result)
    
    def _simulate_transaction(self, order: Order, context: SimulationContext, state_provider: StateProvider) -> SimulationResult:
        """Simulate a single transaction using TxOrder data"""
        try:
            # Ensure we have a TxOrder for transaction simulation
            if order.order_type() != OrderType.TX:
                return SimulationResult(
                    success=False,
                    gas_used=0,
                    coinbase_profit=0,
                    error=SimulationError.UNKNOWN_ERROR,
                    error_message=f"Expected TxOrder, got {order.order_type()}"
                )
            
            # Extract actual transaction data from TxOrder
            tx_order = order  # This is a TxOrder
            from_address = tx_order.sender
            tx_nonce = tx_order.nonce
            max_fee_per_gas = tx_order.max_fee_per_gas
            
            # Decode the actual transaction from raw_tx to get remaining fields
            tx_data = self._decode_transaction_data(tx_order.raw_tx)
            
            # Get account info (with nonce hint for mock provider)
            account = state_provider.get_account(from_address, expected_nonce=tx_nonce)
            if not account:
                return SimulationResult(
                    success=False,
                    gas_used=0,
                    coinbase_profit=0,
                    error=SimulationError.UNKNOWN_ERROR,
                    error_message=f"Account {from_address} not found"
                )
            
            # Check nonce
            if tx_nonce != account.nonce:
                return SimulationResult(
                    success=False,
                    gas_used=0,
                    coinbase_profit=0,
                    error=SimulationError.INVALID_NONCE,
                    error_message=f"Invalid nonce: expected {account.nonce}, got {tx_nonce}"
                )
            
            # Calculate gas costs using actual transaction data
            intrinsic_gas = self._calculate_intrinsic_gas(tx_data.get('to', ''), tx_data.get('data', b''))
            gas_limit = tx_data.get('gas_limit', 21000)
            
            if gas_limit < intrinsic_gas:
                return SimulationResult(
                    success=False,
                    gas_used=0,
                    coinbase_profit=0,
                    error=SimulationError.GAS_LIMIT_EXCEEDED,
                    error_message=f"Gas limit {gas_limit} < intrinsic gas {intrinsic_gas}"
                )
            
            # Check balance for max cost (using actual transaction value and gas)
            value = tx_data.get('value', 0)
            max_cost = value + (max_fee_per_gas * gas_limit)
            if account.balance < max_cost:
                return SimulationResult(
                    success=False,
                    gas_used=0,
                    coinbase_profit=0,
                    error=SimulationError.INSUFFICIENT_BALANCE,
                    error_message=f"Insufficient balance: have {account.balance}, need {max_cost}"
                )
            
            # For simplicity, assume successful execution using intrinsic gas
            # In a real implementation, this would execute the transaction via EVM
            gas_used = intrinsic_gas
            
            # If it's a contract call with data, add some execution gas
            if tx_data.get('to') and tx_data.get('data') and len(tx_data.get('data', b'')) > 0:
                # Estimate additional gas for contract execution
                gas_used += min(gas_limit - intrinsic_gas, 50000)  # Simple heuristic
            
            # Calculate coinbase profit (similar to rbuilder's SimValue)
            coinbase_profit = max_fee_per_gas * gas_used
            
            return SimulationResult(
                success=True,
                gas_used=gas_used,
                coinbase_profit=coinbase_profit,
                state_changes={
                    'from': from_address,
                    'to': tx_data.get('to'),
                    'value': value,
                    'gas_used': gas_used,
                    'gas_price': max_fee_per_gas,
                    'nonce': tx_nonce
                }
            )
            
        except Exception as e:
            logger.error(f"Error in transaction simulation: {e}")
            return SimulationResult(
                success=False,
                gas_used=0,
                coinbase_profit=0,
                error=SimulationError.UNKNOWN_ERROR,
                error_message=str(e)
            )
    
    def _decode_transaction_data(self, raw_tx: bytes) -> Dict[str, Any]:
        """Decode transaction data from raw RLP bytes"""
        try:
            import rlp
            
            first_byte = raw_tx[0]
            
            if first_byte <= 0x7f:  # Typed transaction
                tx_type = first_byte
                payload = raw_tx[1:]
                decoded_payload = rlp.decode(payload)
                
                if tx_type == 0x02:  # EIP-1559
                    fields = decoded_payload
                    return {
                        'to': fields[5].hex() if fields[5] else None,
                        'value': int.from_bytes(fields[6], 'big') if fields[6] else 0,
                        'data': fields[7] if fields[7] else b'',
                        'gas_limit': int.from_bytes(fields[4], 'big') if fields[4] else 21000,
                    }
                elif tx_type == 0x01:  # EIP-2930  
                    fields = decoded_payload
                    return {
                        'to': fields[4].hex() if fields[4] else None,
                        'value': int.from_bytes(fields[5], 'big') if fields[5] else 0,
                        'data': fields[6] if fields[6] else b'',
                        'gas_limit': int.from_bytes(fields[3], 'big') if fields[3] else 21000,
                    }
                elif tx_type == 0x03:  # EIP-4844 Blob
                    # For blob txs, we need the canonical form
                    tx_payload_body = decoded_payload[0]
                    fields = tx_payload_body
                    return {
                        'to': fields[5].hex() if fields[5] else None,
                        'value': int.from_bytes(fields[6], 'big') if fields[6] else 0,
                        'data': fields[7] if fields[7] else b'',
                        'gas_limit': int.from_bytes(fields[4], 'big') if fields[4] else 21000,
                    }
                else:
                    # Unknown typed transaction, use defaults
                    return {
                        'to': None,
                        'value': 0,
                        'data': b'',
                        'gas_limit': 21000,
                    }
            else:  # Legacy transaction
                fields = rlp.decode(raw_tx)
                return {
                    'to': fields[3].hex() if fields[3] else None,
                    'value': int.from_bytes(fields[4], 'big') if fields[4] else 0,
                    'data': fields[5] if fields[5] else b'',
                    'gas_limit': int.from_bytes(fields[2], 'big') if fields[2] else 21000,
                }
                
        except Exception as e:
            logger.warning(f"Failed to decode transaction data: {e}, using defaults")
            # Fallback to safe defaults
            return {
                'to': None,
                'value': 0,
                'data': b'',
                'gas_limit': 21000,
            }
    
    def _calculate_intrinsic_gas(self, to_address: str, data) -> int:
        """Calculate intrinsic gas cost for a transaction"""
        gas = 21000  # Base transaction cost
        
        # Add cost for data
        if data:
            if isinstance(data, bytes):
                data_bytes = data
            elif isinstance(data, str) and data != '0x' and data != '':
                # Remove 0x prefix if present
                hex_data = data[2:] if data.startswith('0x') else data
                try:
                    data_bytes = bytes.fromhex(hex_data)
                except ValueError:
                    data_bytes = b''  # Invalid hex, treat as empty
            else:
                data_bytes = b''
            
            for byte in data_bytes:
                if byte == 0:
                    gas += 4  # Zero byte cost
                else:
                    gas += 16  # Non-zero byte cost
        
        # Add cost for contract creation
        if not to_address or to_address == '':
            gas += 32000  # Contract creation cost
        
        return gas
    
    def _simulate_bundle(self, order: Order, context: SimulationContext, state_provider: StateProvider) -> SimulationResult:
        """Simulate a bundle order - similar to rbuilder's bundle simulation logic"""
        try:
            if not hasattr(order, 'child_orders'):
                return SimulationResult(
                    success=False,
                    gas_used=0,
                    coinbase_profit=0,
                    error=SimulationError.UNKNOWN_ERROR,
                    error_message="Bundle order missing child_orders attribute"
                )
            
            total_gas_used = 0
            total_coinbase_profit = 0
            successful_txs = 0
            state_changes = []
            
            # Simulate each transaction in the bundle
            # For bundles: at least one tx must succeed, mandatory txs must all succeed
            for child_order, optional in order.child_orders:
                child_result = self._simulate_child_order(child_order, context, state_provider)
                    
                if child_result.success:
                    total_gas_used += child_result.gas_used
                    total_coinbase_profit += child_result.coinbase_profit
                    successful_txs += 1
                    if child_result.state_changes:
                        state_changes.append(child_result.state_changes)
                else:
                    # If a mandatory transaction fails, the whole bundle fails
                    if not optional:
                        return SimulationResult(
                            success=False,
                            gas_used=0,
                            coinbase_profit=0,
                            error=child_result.error,
                            error_message=f"Mandatory tx failed: {child_result.error_message}",
                            state_changes={'failed_tx': child_result.state_changes}
                        )
            
            # Bundle succeeds if at least one transaction succeeded
            success = successful_txs > 0
            
            return SimulationResult(
                success=success,
                gas_used=total_gas_used,
                coinbase_profit=total_coinbase_profit,
                state_changes={
                    'bundle_type': 'bundle',
                    'bundle_txs': state_changes,
                    'successful_txs': successful_txs,
                    'total_txs': len(order.child_orders)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in bundle simulation: {e}")
            return SimulationResult(
                success=False,
                gas_used=0,
                coinbase_profit=0,
                error=SimulationError.UNKNOWN_ERROR,
                error_message=str(e)
            )
    
    def _simulate_share_bundle(self, order: Order, context: SimulationContext, state_provider: StateProvider) -> SimulationResult:
        """Simulate a share bundle order - stricter rules than regular bundles"""
        try:
            if not hasattr(order, 'child_orders'):
                return SimulationResult(
                    success=False,
                    gas_used=0,
                    coinbase_profit=0,
                    error=SimulationError.UNKNOWN_ERROR,
                    error_message="Share bundle order missing child_orders attribute"
                )
            
            total_gas_used = 0
            total_coinbase_profit = 0
            successful_txs = 0
            state_changes = []
            
            # Simulate each transaction in the share bundle
            # For share bundles: ALL transactions must succeed (stricter than bundles)
            for child_order, optional in order.child_orders:
                child_result = self._simulate_child_order(child_order, context, state_provider)
                    
                if child_result.success:
                    total_gas_used += child_result.gas_used
                    total_coinbase_profit += child_result.coinbase_profit
                    successful_txs += 1
                    if child_result.state_changes:
                        state_changes.append(child_result.state_changes)
                else:
                    # For share bundles, ANY failure means total failure
                    # (unless the transaction is explicitly marked as optional)
                    if not optional:
                        return SimulationResult(
                            success=False,
                            gas_used=0,
                            coinbase_profit=0,
                            error=child_result.error,
                            error_message=f"Share bundle tx failed: {child_result.error_message}",
                            state_changes={'failed_tx': child_result.state_changes}
                        )
            
            # Share bundle succeeds only if all non-optional transactions succeeded
            success = successful_txs > 0
            
            return SimulationResult(
                success=success,
                gas_used=total_gas_used,
                coinbase_profit=total_coinbase_profit,
                state_changes={
                    'bundle_type': 'share_bundle',
                    'share_bundle_txs': state_changes,
                    'successful_txs': successful_txs,
                    'total_txs': len(order.child_orders)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in share bundle simulation: {e}")
            return SimulationResult(
                success=False,
                gas_used=0,
                coinbase_profit=0,
                error=SimulationError.UNKNOWN_ERROR,
                error_message=str(e)
            )
    
    def _simulate_child_order(self, child_order: Order, context: SimulationContext, state_provider: StateProvider) -> SimulationResult:
        """Simulate a child order (transaction or nested bundle)"""
        if child_order.order_type() == OrderType.TX:
            return self._simulate_transaction(child_order, context, state_provider)
        elif child_order.order_type() == OrderType.BUNDLE:
            return self._simulate_bundle(child_order, context, state_provider)
        elif child_order.order_type() == OrderType.SHAREBUNDLE:
            return self._simulate_share_bundle(child_order, context, state_provider)
        else:
            return SimulationResult(
                success=False,
                gas_used=0,
                coinbase_profit=0,
                error=SimulationError.UNKNOWN_ERROR,
                error_message=f"Unknown child order type: {child_order.order_type()}"
            )
    
    def simulate_orders(self, orders: List[Order], context: SimulationContext) -> List[SimulatedOrder]:
        """Simulate multiple orders"""
        results = []
        for order in orders:
            result = self.simulate_order(order, context)
            results.append(result)
        return results


def simulate_orders_with_mock_provider(orders: List[Order], block_number: int) -> List[SimulatedOrder]:
    """
    Convenience function for simulating orders with mock state provider.
    """
    from .mock_provider import MockStateProviderFactory
    
    # Create mock state provider factory
    factory = MockStateProviderFactory(block_number)
    
    # Get simulation context
    context = factory.get_simulation_context(block_number)
    
    # Create simulator
    simulator = SimpleOrderSimulator(factory)
    
    # Simulate orders
    return simulator.simulate_orders(orders, context)
