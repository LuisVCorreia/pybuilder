from dataclasses import dataclass
from typing import Dict, List, Tuple
from .sim_utils import SimulatedOrder
from backtest.common.order import Order

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
class SimulationRequest:
    """A request to simulate an order with its parents."""
    order: Order
    parents: List[Order]

@dataclass
class SimulatedResult:
    """The result of a simulation, used to wake up pending orders."""
    simulated_order: SimulatedOrder
    parents: List[Order]
    nonces_after: List[NonceKey]

class SimTree:
    def __init__(self, on_chain_nonces: Dict[str, int]):
        self.on_chain_nonces = on_chain_nonces.copy()
        self.sims_by_resulting_nonce: Dict[NonceKey, SimulatedResult] = {}
        self.pending_orders: Dict[str, Tuple[Order, int]] = {}
        self.pending_nonces: Dict[NonceKey, List[str]] = {}
        self.ready_queue: List[SimulationRequest] = []

    def _get_onchain_nonce(self, address: str) -> int:
        return self.on_chain_nonces.get(address, 0)

    def _get_order_nonce_state(self, order: Order) -> Tuple[str, List[Order] | List[NonceKey]]:
        parents = []
        pending_nonces_for_this_order = []

        for tx_nonce in order.nonces():
            on_chain_nonce = self._get_onchain_nonce(tx_nonce.address)
            required_nonce = tx_nonce.nonce

            if required_nonce < on_chain_nonce:
                if not tx_nonce.optional: return ("Invalid", [])
                continue
            
            if required_nonce > on_chain_nonce:
                key_to_find = NonceKey(tx_nonce.address, required_nonce)

                if key_to_find in self.sims_by_resulting_nonce:
                    # Found a simulation that can act as a parent.
                    parent_sim_result = self.sims_by_resulting_nonce[key_to_find]
                    
                    parents.extend(parent_sim_result.parents)
                    parents.append(parent_sim_result.simulated_order.order)
                else:
                    # No parent sim found, so we are pending on this nonce.
                    pending_nonces_for_this_order.append(key_to_find)

        if not pending_nonces_for_this_order:
            # We must de-duplicate and sort the final parent list.
            # Using dict.fromkeys preserves order and gives unique items.
            unique_parents = list(dict.fromkeys(parents))
            return ("Ready", unique_parents)
        else:
            return ("Pending", pending_nonces_for_this_order)

    def push_order(self, order: Order):
        order_id = order.id()
        if order_id in self.pending_orders: return

        status, data = self._get_order_nonce_state(order)

        if status == "Ready":
            self.ready_queue.append(SimulationRequest(order=order, parents=data))
        elif status == "Pending":
            pending_nonces = data
            self.pending_orders[order_id] = (order, len(pending_nonces))
            for nonce_key in pending_nonces:
                self.pending_nonces.setdefault(nonce_key, []).append(order_id)

    def pop_simulation_requests(self, limit: int) -> List[SimulationRequest]:
        limit = min(limit, len(self.ready_queue))
        return [self.ready_queue.pop(0) for _ in range(limit)]

    def submit_simulation_result(self, result: SimulatedResult):
        # A sim of nonce N results in account nonce N+1.
        # This is the key for both storing the result and waking up pending orders.
        for nonce_key_after in result.nonces_after:
            self.sims_by_resulting_nonce[nonce_key_after] = result

            # Wake up any orders that were pending on this exact resulting nonce.
            if nonce_key_after in self.pending_nonces:
                for order_id in self.pending_nonces.pop(nonce_key_after, []):
                    if order_id in self.pending_orders:
                        order, unsatisfied = self.pending_orders[order_id]
                        new_unsatisfied = unsatisfied - 1

                        if new_unsatisfied <= 0:
                            self.pending_orders.pop(order_id)
                            # Re-check the order; it's now ready.
                            self.push_order(order)
                        else:
                            self.pending_orders[order_id] = (order, new_unsatisfied)
