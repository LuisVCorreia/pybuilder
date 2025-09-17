from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum
from backtest.common.order import Order
from .state_trace import UsedStateTrace
from decimal import Decimal

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
    prevrandao: Optional[str] = None
    uncles_hash: Optional[str] = None
    state_root: Optional[str] = None
    transaction_root: Optional[str] = None
    receipt_root: Optional[str] = None
    nonce: Optional[str] = None
    mix_hash: Optional[str] = None
    parent_beacon_block_root: Optional[str] = None
    bloom: Optional[str] = None
    extra_data: Optional[str] = None
    requests_hash: Optional[str] = None

    @classmethod
    def from_onchain_block(cls, onchain_block: dict, winning_bid_trace: dict = None) -> 'SimulationContext':
        """
        Create SimulationContext from onchain block data.
        """

        # Extract all the fields we need for proper block header construction
        context = cls(
            block_number=onchain_block.get('number'),
            block_timestamp=onchain_block.get('timestamp'),
            block_base_fee=onchain_block.get('baseFeePerGas'),
            block_gas_limit=onchain_block.get('gasLimit'),
            block_hash=onchain_block.get('hash'),
            parent_hash=onchain_block.get('parentHash'),
            chain_id=1,  # Ethereum mainnet
            coinbase=onchain_block.get('miner'),
            block_difficulty=onchain_block.get('difficulty'),
            block_gas_used=onchain_block.get('gasUsed'),
            withdrawals_root=onchain_block.get('withdrawalsRoot'),
            blob_gas_used=onchain_block.get('blobGasUsed'),
            excess_blob_gas=onchain_block.get('excessBlobGas'),
            prevrandao=onchain_block.get('prevRandao'),
            uncles_hash=onchain_block.get('sha3Uncles'),
            state_root=onchain_block.get('stateRoot'),
            transaction_root=onchain_block.get('transactionsRoot'),
            receipt_root=onchain_block.get('receiptsRoot'),
            nonce=onchain_block.get('nonce'),
            mix_hash=onchain_block.get('mixHash'),
            parent_beacon_block_root=onchain_block.get('parentBeaconBlockRoot'),
            bloom=onchain_block.get('logsBloom'),
            extra_data=onchain_block.get('extraData'),
            requests_hash=onchain_block.get('requestsHash')
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
    """Internal result of simulating an order"""
    success: bool
    gas_used: int
    coinbase_profit: int = 0  # in wei
    blob_gas_used: int = 0
    paid_kickbacks: int = 0  # simplified - just total value in wei
    error: Optional[SimulationError] = None
    error_message: Optional[str] = None
    state_changes: Optional[Dict[str, Any]] = None
    state_trace: Optional[UsedStateTrace] = None  # State tracing information


@dataclass
class SimValue:
    """Economic value of a simulation, matching rbuilder's SimValue"""
    coinbase_profit: int  # in wei
    gas_used: int
    blob_gas_used: int
    mev_gas_price: Decimal
    paid_kickbacks: int  # in wei


@dataclass
class SimulatedOrder:
    """An order with its simulation results, matching rbuilder's SimulatedOrder"""
    order: Order
    sim_value: SimValue
    used_state_trace: Optional[UsedStateTrace] = None  # State tracing information
    _error_result: Optional[OrderSimResult] = None  # For failed orders
    sim_duration: float = 0.0  # Duration of the simulation

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
                paid_kickbacks=self.sim_value.paid_kickbacks,
                state_trace=self.used_state_trace
            )

    def serialize(self) -> dict:
        """Converts a SimulatedOrder object to a JSON-serializable dictionary."""
        return {
            "order_id": str(self.order.id()),
            "gas_used": self.sim_value.gas_used,
            "coinbase_profit": self.sim_value.coinbase_profit,
            "blob_gas_used": self.sim_value.blob_gas_used,
            "state_trace": self.used_state_trace.serialize()
        }
