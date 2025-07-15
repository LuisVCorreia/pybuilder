import heapq
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Optional
from collections import defaultdict

from backtest.build.simulation.sim_utils import SimulatedOrder, SimValue
from backtest.common.order import OrderId, TxNonce

logger = logging.getLogger(__name__)

# Minimum simulation result percentage (95% like rbuilder)
MIN_SIM_RESULT_PERCENTAGE = 95


class Sorting(Enum):
    """
    Sorting criteria for ordering simulated orders.
    """
    MAX_PROFIT = "max-profit"
    MEV_GAS_PRICE = "mev-gas-price"

    @classmethod
    def from_str(cls, value: str) -> 'Sorting':
        """Parse sorting from string value."""
        for sorting in cls:
            if sorting.value == value:
                return sorting
        raise ValueError(f"Invalid sorting: {value}")


@dataclass(frozen=True)
class AccountNonce:
    """Represents an account nonce pair. Equivalent to TxNonce but for internal use."""
    account: str
    nonce: int


class OrderPriority(ABC):
    """
    Abstract base class for order priority implementations.
    """
    
    def __init__(self, order: SimulatedOrder):
        self.order = order
    
    @abstractmethod
    def priority_value(self) -> int:
        """Return the priority value for heap ordering (higher = better priority)."""
        pass
    
    @classmethod
    @abstractmethod
    def simulation_too_low(cls, original_sim_value: SimValue, new_sim_value: SimValue) -> bool:
        """Check if new simulation result is too low compared to original."""
        pass
    
    def __lt__(self, other: 'OrderPriority') -> bool:
        """
        Comparison for heap ordering. Python's heapq is a min-heap,
        so we negate the priority to get max-heap behavior.
        """
        if self.priority_value() != other.priority_value():
            return self.priority_value() > other.priority_value()  # Higher priority first
        # Break ties with order ID for deterministic ordering (matches rbuilder's logic)
        # Since we're inverting for max-heap behavior, we also invert the tie-breaker
        return self.order.order.id() > other.order.order.id()


class MaxProfitPriority(OrderPriority):
    """
    Orders by absolute coinbase profit.
    """
    
    def priority_value(self) -> int:
        return self.order.sim_value.coinbase_profit
    
    @classmethod
    def simulation_too_low(cls, original_sim_value: SimValue, new_sim_value: SimValue) -> bool:
        """Check if profit dropped too much (below 95% of original)."""
        return cls._new_sim_value_too_low(
            original_sim_value.coinbase_profit,
            new_sim_value.coinbase_profit
        )
    
    @staticmethod
    def _new_sim_value_too_low(original_sim: int, new_sim: int) -> bool:
        """Generic function for checking if simulation result is too low."""
        if original_sim == 0:
            return False
        return (new_sim * 100) < (original_sim * MIN_SIM_RESULT_PERCENTAGE)


class MevGasPricePriority(OrderPriority):
    """
    Orders by MEV gas price (coinbase_profit / gas_used).
    """
    
    def priority_value(self) -> int:
        """Calculate MEV gas price."""
        if self.order.sim_value.gas_used == 0:
            return 0
        return self.order.sim_value.coinbase_profit // self.order.sim_value.gas_used
    
    @classmethod
    def simulation_too_low(cls, original_sim_value: SimValue, new_sim_value: SimValue) -> bool:
        """Check if MEV gas price dropped too much."""
        original_price = cls._mev_gas_price(original_sim_value)
        new_price = cls._mev_gas_price(new_sim_value)
        return MaxProfitPriority._new_sim_value_too_low(original_price, new_price)
    
    @staticmethod
    def _mev_gas_price(sim_value: SimValue) -> int:
        """Calculate MEV gas price from SimValue."""
        if sim_value.gas_used == 0:
            return 0
        return sim_value.coinbase_profit // sim_value.gas_used


class PrioritizedOrderStore:
    """
    Heap-based order store that manages orders by priority and tracks nonce dependencies.
    """
    
    def __init__(self, priority_class: type, initial_nonces: Optional[List[TxNonce]] = None):
        self.priority_class = priority_class
        
        # Main heap for ready-to-execute orders (min-heap, but we negate priorities)
        self.main_queue: List[OrderPriority] = []
        
        # Track which accounts each order affects for nonce invalidation
        self.main_queue_nonces: Dict[str, List[OrderId]] = defaultdict(list)
        
        # Current on-chain nonces for each account
        self.onchain_nonces: Dict[str, int] = {}
        if initial_nonces:
            for nonce in initial_nonces:
                self.onchain_nonces[nonce.address] = nonce.nonce
        
        # Orders waiting for nonce dependencies
        self.pending_orders: Dict[AccountNonce, List[OrderId]] = defaultdict(list)
        
        # All orders we manage (by OrderId)
        self.orders: Dict[OrderId, SimulatedOrder] = {}
    
    def insert_order(self, sim_order: SimulatedOrder, force_reinsert: bool = False) -> None:
        order_id = sim_order.order.id()
        
        # Don't insert if we already have this order (unless forcing reinsert)
        if order_id in self.orders and not force_reinsert:
            logger.debug(f"Order {order_id} already in store, skipping")
            return
        
        # Check nonce dependencies
        pending_nonces = []
        for nonce_info in sim_order.order.nonces():
            account = nonce_info.address
            required_nonce = nonce_info.nonce
            optional = nonce_info.optional
            
            current_nonce = self.onchain_nonces.get(account, 0)
            
            logger.debug(f"Order {order_id}: account {account}, required nonce {required_nonce}, current {current_nonce}, optional {optional}")
            
            # Order is invalid if required nonce is too old
            if current_nonce > required_nonce and not optional:
                logger.debug(f"Order {order_id} invalid: nonce {required_nonce} < current {current_nonce}")
                return
            
            # Order must wait if required nonce is in the future
            if current_nonce < required_nonce and not optional:
                logger.debug(f"Order {order_id} pending: needs nonce {required_nonce}, current {current_nonce}")
                pending_nonces.append(AccountNonce(account=account, nonce=required_nonce))
        
        # Store the order
        self.orders[order_id] = sim_order
        
        if pending_nonces:
            # Order must wait for nonce dependencies
            logger.debug(f"Order {order_id} waiting for {len(pending_nonces)} pending nonces")
            for pending_nonce in pending_nonces:
                self.pending_orders[pending_nonce].append(order_id)
        else:
            # Order is ready to execute
            logger.debug(f"Order {order_id} ready to execute, adding to main queue")
            priority = self.priority_class(sim_order)
            heapq.heappush(self.main_queue, priority)
            
            # Track which accounts this order affects
            for nonce_info in sim_order.order.nonces():
                self.main_queue_nonces[nonce_info.address].append(order_id)
    
    def pop_order(self) -> Optional[SimulatedOrder]:
        """
        Pop the highest priority order from the main queue.
        """
        if not self.main_queue:
            return None
        
        priority = heapq.heappop(self.main_queue)
        order_id = priority.order.order.id()
        
        # Clean up tracking data
        order = self._remove_popped_order(order_id)
        return order
    
    def _remove_popped_order(self, order_id: OrderId) -> Optional[SimulatedOrder]:
        """Clean up after an order was removed from main_queue."""
        sim_order = self.orders.pop(order_id, None)
        if not sim_order:
            return None
        
        # Remove from nonce tracking
        for nonce_info in sim_order.order.nonces():
            account = nonce_info.address
            if account in self.main_queue_nonces:
                try:
                    self.main_queue_nonces[account].remove(order_id)
                    if not self.main_queue_nonces[account]:
                        del self.main_queue_nonces[account]
                except ValueError:
                    pass  # Order wasn't in the list
        
        return sim_order
    
    def update_onchain_nonces(self, new_nonces: List[TxNonce]) -> None:
        """
        Update on-chain nonces and process newly available orders.
        """
        invalidated_orders: Set[OrderId] = set()
        
        # Update nonces and collect invalidated orders
        for new_nonce in new_nonces:
            account = new_nonce.address
            self.onchain_nonces[account] = new_nonce.nonce
            
            # Collect orders that might be invalidated
            if account in self.main_queue_nonces:
                for order_id in self.main_queue_nonces[account]:
                    invalidated_orders.add(order_id)
        
        # Process invalidated orders
        valid_orders = []
        for order_id in invalidated_orders:
            # Remove from main queue (will be re-added if still valid)
            self._remove_from_main_queue(order_id)
            
            # Get the order and check if it's still valid
            order = self.orders.get(order_id)
            if order:
                valid_nonces = 0
                is_valid = True
                
                for nonce_info in order.order.nonces():
                    account = nonce_info.address
                    required_nonce = nonce_info.nonce
                    optional = nonce_info.optional
                    
                    current_nonce = self.onchain_nonces.get(account, 0)
                    
                    if current_nonce > required_nonce and not optional:
                        is_valid = False
                        break
                    elif current_nonce == required_nonce:
                        valid_nonces += 1
                
                # Re-insert valid orders
                if is_valid and valid_nonces > 0:
                    valid_orders.append(order)
                else:
                    # Remove invalid orders
                    self.orders.pop(order_id, None)
        
        # Re-insert valid orders
        for order in valid_orders:
            self.insert_order(order)
        
        # Process newly available pending orders
        for new_nonce in new_nonces:
            pending_key = AccountNonce(account=new_nonce.address, nonce=new_nonce.nonce)
            logger.debug(f"Checking for pending orders with key {pending_key}")
            if pending_key in self.pending_orders:
                pending_order_ids = self.pending_orders.pop(pending_key)
                logger.debug(f"Found {len(pending_order_ids)} pending orders for {pending_key}")
                for order_id in pending_order_ids:
                    order = self.orders.get(order_id)
                    if order:
                        logger.debug(f"Making pending order {order_id} available")
                        self.insert_order(order, force_reinsert=True)
            else:
                logger.debug(f"No pending orders found for {pending_key}")
    
    def _remove_from_main_queue(self, order_id: OrderId) -> None:
        """Remove an order from the main queue by order ID."""
        # Since we can't efficiently remove from middle of heap,
        # we mark as removed and filter during pop
        new_queue = []
        for priority in self.main_queue:
            if priority.order.order.id() != order_id:
                new_queue.append(priority)
        
        self.main_queue = new_queue
        heapq.heapify(self.main_queue)
    
    def remove_order(self, order_id: OrderId) -> Optional[SimulatedOrder]:
        """Remove an order by ID from the store."""
        # Remove from main queue if present
        self._remove_from_main_queue(order_id)
        
        # Remove from tracking and return
        return self.orders.pop(order_id, None)
    
    def get_all_orders(self) -> List[SimulatedOrder]:
        """Get all orders currently in the store."""
        return list(self.orders.values())
    
    def is_empty(self) -> bool:
        """Check if the main queue is empty."""
        return len(self.main_queue) == 0
    
    def size(self) -> int:
        """Get the number of orders in the main queue."""
        return len(self.main_queue)


def create_priority_class(sorting: Sorting) -> type:
    """
    Factory function to create the appropriate priority class.
    """
    if sorting == Sorting.MAX_PROFIT:
        return MaxProfitPriority
    elif sorting == Sorting.MEV_GAS_PRICE:
        return MevGasPricePriority
    else:
        raise ValueError(f"Unsupported sorting: {sorting}")
