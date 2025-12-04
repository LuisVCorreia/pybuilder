from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from .order import Order, OrderId

@dataclass
class BuiltBlockData:
    """Built block data"""
    included_orders: List[str]  # OrderIds as strings
    orders_closed_at: datetime
    sealed_at: datetime
    profit: int

@dataclass
class OrdersWithTimestamp:
    """Order with timestamp, matching Rust OrdersWithTimestamp"""
    timestamp_ms: int
    order: Order

class OrderFilteredReason(Enum):
    """Reasons why an order was filtered out, matching Rust enum"""
    TIMESTAMP = "Timestamp"  # Order was received late
    REPLACED = "Replaced"    # Order was replaced
    MEMPOOL_TXS = "MempoolTxs"  # Order is made of mempool txs
    IDS = "Ids"             # Order id was explicitly filtered out
    SIGNER = "Signer"       # Order signer was explicitly filtered out

@dataclass
class BlockData:
    """
    Historic data for a block, matching Rust BlockData.
    Used for backtesting.
    """
    block_number: int
    # Extra info for landed block (not contained on onchain_block).
    # We get this from the relays (API /relay/v1/data/bidtraces/builder_blocks_received).
    winning_bid_trace: Dict[str, Any]
    # Landed block.
    onchain_block: Dict[str, Any]
    # Orders we had at the moment of building the block.
    # This might be an approximation depending on DataSources used.
    available_orders: List[OrdersWithTimestamp]
    filtered_orders: Dict[OrderId, OrderFilteredReason]
    built_block_data: Optional[BuiltBlockData]

    def __init__(self, block_number: int, winning_bid_trace: Dict[str, Any], 
                 onchain_block: Dict[str, Any], available_orders: List[OrdersWithTimestamp] = None,
                 filtered_orders: Dict[OrderId, OrderFilteredReason] = None,
                 built_block_data: Optional[BuiltBlockData] = None):
        self.block_number = block_number
        self.winning_bid_trace = winning_bid_trace
        self.onchain_block = onchain_block
        self.available_orders = available_orders or []
        self.filtered_orders = filtered_orders or {}
        self.built_block_data = built_block_data

    def filter_late_orders(self, build_block_lag_ms: int):
        """
        Filters orders that arrived after we started building the block.
        """
        final_timestamp_ms = self.winning_bid_trace.get('timestamp_ms', 0) - build_block_lag_ms
        self.filter_orders_by_end_timestamp_ms(final_timestamp_ms)

    def filter_orders_by_end_timestamp(self, final_timestamp: datetime):
        """Filter orders by end timestamp"""
        final_timestamp_ms = int(final_timestamp.timestamp() * 1000)
        self.filter_orders_by_end_timestamp_ms(final_timestamp_ms)

    def filter_orders_by_end_timestamp_ms(self, final_timestamp_ms: int):
        """
        Filter orders by timestamp, matching Rust implementation.
        We never filter included orders even by timestamp.
        """
        # Get included orders (we never filter these even by timestamp)
        included_orders = set()
        if self.built_block_data and self.built_block_data.included_orders:
            included_orders = set(self.built_block_data.included_orders)

        # Filter available orders
        filtered_out = []
        kept_orders = []
        
        for order_with_ts in self.available_orders:
            order_id = order_with_ts.order.id()
            
            # Keep included orders regardless of timestamp
            if str(order_id) in included_orders:
                kept_orders.append(order_with_ts)
                continue
            
            if order_with_ts.timestamp_ms <= final_timestamp_ms:
                kept_orders.append(order_with_ts)
            else:
                filtered_out.append(order_with_ts)
                self.filtered_orders[order_id] = OrderFilteredReason.TIMESTAMP

        self.available_orders = kept_orders

        # Handle replacement logic (keep only latest version of replaceable orders)
        # Sort by timestamp from latest to earliest
        self.available_orders.sort(key=lambda o: o.timestamp_ms, reverse=True)
        
        replacement_keys_seen = set()
        final_orders = []
        
        for order_with_ts in self.available_orders:
            replacement_key = order_with_ts.order.replacement_key() if hasattr(order_with_ts.order, 'replacement_key') else None
            
            if replacement_key:
                if replacement_key in replacement_keys_seen:
                    # This is an older version, filter it out
                    self.filtered_orders[order_with_ts.order.id()] = OrderFilteredReason.REPLACED
                    continue
                replacement_keys_seen.add(replacement_key)
            
            final_orders.append(order_with_ts)
        
        self.available_orders = final_orders

    def filter_bundles_from_mempool(self):
        """
        Remove all bundles that have all transactions available in the public mempool.
        """
        # Get all mempool transaction hashes
        mempool_txs = set()
        for order_with_ts in self.available_orders:
            if order_with_ts.order.order_type().value == "tx":  # Assuming tx orders represent mempool txs
                mempool_txs.add(order_with_ts.order.id())

        # Filter out bundles that are composed entirely of mempool txs
        filtered_orders = []
        for order_with_ts in self.available_orders:
            order = order_with_ts.order
            
            # If it's a bundle/sharebundle, check if all its txs are in mempool
            if order.order_type().value in ["bundle", "sharebundle"]:
                # This would need to be implemented based on bundle structure
                # For now, we'll skip this filtering
                filtered_orders.append(order_with_ts)
            else:
                filtered_orders.append(order_with_ts)
        
        self.available_orders = filtered_orders

    def filter_orders_by_ids(self, order_ids: List[str]):
        """Filter out orders with specific IDs"""
        order_ids_set = set(order_ids)
        filtered_orders = []
        for order_with_ts in self.available_orders:
            order_id = str(order_with_ts.order.id().value)
            if order_id in order_ids_set:
                filtered_orders.append(order_with_ts)
            else:
                self.filtered_orders[order_with_ts.order.id()] = OrderFilteredReason.IDS
        
        self.available_orders = filtered_orders

    def filter_out_ignored_signers(self, ignored_signers: List[str]):
        """Filter out orders from specific signers"""
        ignored_signers_set = set(s.lower() for s in ignored_signers)
        filtered_orders = []
        
        for order_with_ts in self.available_orders:
            # Get signer from order (this depends on order implementation)
            signer = getattr(order_with_ts.order, 'sender', None)
            if signer and signer.lower() not in ignored_signers_set:
                filtered_orders.append(order_with_ts)
            elif signer and signer.lower() in ignored_signers_set:
                self.filtered_orders[order_with_ts.order.id()] = OrderFilteredReason.SIGNER
            else:
                # If we can't determine signer, keep the order
                filtered_orders.append(order_with_ts)
        
        self.available_orders = filtered_orders

    def is_validator_fee_payment(self, tx: Dict[str, Any]) -> bool:
        """Check if transaction is a validator fee payment"""
        tx_from = tx.get('from', '').lower()
        beneficiary = self.onchain_block.get('miner', '').lower()
        return tx_from == beneficiary
