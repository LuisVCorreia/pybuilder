import enum
import uuid
from dataclasses import dataclass
from typing import Any, List, Union, Dict, Tuple
import rlp
from rlp.exceptions import DecodingError
import pandas as pd
import logging

from web3 import Web3
from ..fetch.root_provider import RootProvider
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import json

from eth.abc import SignedTransactionAPI
from eth.vm.forks.prague.transactions import SetCodeTransaction, PragueTypedTransaction, PragueLegacyTransaction
from eth.vm.forks.berlin.transactions import AccessListTransaction
from eth.vm.forks.london.transactions import DynamicFeeTransaction
from eth.vm.forks.cancun.transactions import BlobTransaction

logger = logging.getLogger(__name__)


def safe_int_from_field(field) -> int:
    """Safely convert RLP field to int, handling bytes and int."""
    if isinstance(field, bytes):
        return int.from_bytes(field, 'big') if field else 0
    if isinstance(field, int):
        return field
    logger.warning(f"safe_int_from_field received unexpected type: {type(field)}, value: {field}")
    return 0


@dataclass
class TxNonce:
    """Represents a transaction nonce with address and optional flag."""
    address: str
    nonce: int
    optional: bool

class OrderType(enum.Enum):
    TX = "tx"
    BUNDLE = "bundle"
    SHAREBUNDLE = "sbundle"

@dataclass(frozen=True)
class OrderId:
    """
    Formats:
      - tx:0x<hash>
      - bundle:<uuid>
      - sbundle:0x<hash>
    """
    type: OrderType
    value: str

    def __str__(self) -> str:
        return f"{self.type.value}:{self.value}"
    
    def fixed_bytes(self) -> bytes:
        """
        Convert OrderId to fixed 32-byte representation for comparison.
        """
        if self.type in (OrderType.TX, OrderType.SHAREBUNDLE):
            # For tx and sharebundle, use the hash directly
            hex_str = self.value[2:] if self.value.startswith('0x') else self.value
            # Pad to 64 hex characters (32 bytes) if needed
            hex_str = hex_str.zfill(64)
            return bytes.fromhex(hex_str)
        elif self.type == OrderType.BUNDLE:
            # For bundle, convert UUID to bytes and pad to 32 bytes
            import uuid
            bundle_uuid = uuid.UUID(self.value)
            uuid_bytes = bundle_uuid.bytes  # 16 bytes
            # Pad with zeros to make 32 bytes (like Rust implementation)
            return uuid_bytes + b'\x00' * 16
        else:
            raise ValueError(f"Unknown OrderType: {self.type}")
    
    def _rank(self) -> int:
        """Get the rank for tie-breaking."""
        if self.type == OrderType.TX:
            return 1
        elif self.type == OrderType.BUNDLE:
            return 2
        elif self.type == OrderType.SHAREBUNDLE:
            return 3
        else:
            raise ValueError(f"Unknown OrderType: {self.type}")
    
    def __lt__(self, other: 'OrderId') -> bool:
        """Compare OrderIds using fixed bytes first, then rank."""
        if not isinstance(other, OrderId):
            return NotImplemented
        
        # First compare fixed bytes
        self_bytes = self.fixed_bytes()
        other_bytes = other.fixed_bytes()
        
        if self_bytes != other_bytes:
            return self_bytes < other_bytes
        
        # If bytes are equal, compare by rank
        return self._rank() < other._rank()
    
    def __le__(self, other: 'OrderId') -> bool:
        return self == other or self < other
    
    def __gt__(self, other: 'OrderId') -> bool:
        return not self <= other
    
    def __ge__(self, other: 'OrderId') -> bool:
        return not self < other

    @classmethod
    def from_tx(cls, tx_hash: Union[str, bytes]) -> 'OrderId':
        hex_str = tx_hash.hex() if isinstance(tx_hash, bytes) else tx_hash.lower().lstrip('0x')
        return cls(type=OrderType.TX, value="0x" + hex_str)

    @classmethod
    def from_bundle(cls, bundle_uuid: uuid.UUID) -> 'OrderId':
        return cls(type=OrderType.BUNDLE, value=str(bundle_uuid))

    @classmethod
    def from_sharebundle(cls, sb_hash: Union[str, bytes]) -> 'OrderId':
        hex_str = sb_hash.hex() if isinstance(sb_hash, bytes) else sb_hash.lower().lstrip('0x')
        return cls(type=OrderType.SHAREBUNDLE, value="0x" + hex_str)

    @classmethod
    def parse(cls, text: str) -> 'OrderId':
        try:
            t, v = text.split(':', 1)
            otype = OrderType(t)
        except Exception:
            raise ValueError(f"Invalid OrderId string: {text}")
        if otype == OrderType.BUNDLE:
            u = uuid.UUID(v)
            return cls(type=otype, value=str(u))
        val = v if v.startswith('0x') else '0x' + v
        return cls(type=otype, value=val)

    def to_bytes(self) -> bytes:
        if self.type in (OrderType.TX, OrderType.SHAREBUNDLE):
            return bytes.fromhex(self.value[2:])
        raise TypeError(f"OrderId {self.type} cannot be converted to bytes")


class Order:
    """
    Base class for all orders: Transaction, Bundle, or ShareBundle.
    """
    def id(self) -> OrderId:
        raise NotImplementedError

    def order_type(self) -> OrderType:
        raise NotImplementedError

    def order_data(self) -> Dict[str, Any]:
        raise NotImplementedError

    def can_execute_with_block_base_fee(self, block_base_fee: int) -> bool:
        raise NotImplementedError
    
    def nonces(self) -> List[TxNonce]:
        """Return list of nonces this order depends on."""
        raise NotImplementedError

    def transactions(self) -> List[Dict[str, Any]]:
        """Return list of transactions in this order."""
        raise NotImplementedError

    @classmethod
    def from_serialized(cls, order_type_str: str, serialized_data: bytes) -> 'Order':
        """
        Factory method to deserialize an order from its database representation.
        """
        order_data = json.loads(serialized_data.decode('utf-8'))
        order_type = OrderType(order_type_str)

        if order_type == OrderType.TX:
            raw_tx_hex = order_data['raw_tx']
            if raw_tx_hex.startswith('0x'):
                raw_tx_hex = raw_tx_hex[2:]
            raw_tx = bytes.fromhex(raw_tx_hex)
            
            # We can re-derive the full TxOrder by using the from_raw constructor
            return TxOrder.from_raw(order_data['timestamp_ms'], raw_tx)
        
        elif order_type == OrderType.BUNDLE:
            # Reconstruct inner orders
            inner_orders = [
                cls.from_serialized(o['order_type'], json.dumps(o['data']).encode('utf-8')) 
                for o in order_data['orders']
            ]
            return BundleOrder(
                order_data['timestamp_ms'],
                order_data['bundle_id'],
                inner_orders
            )

        elif order_type == OrderType.SHAREBUNDLE:
            # Reconstruct inner orders
            inner_orders = [
                cls.from_serialized(o['order_type'], json.dumps(o['data']).encode('utf-8'))
                for o in order_data['orders']
            ]
            return ShareBundleOrder(
                order_data['timestamp_ms'],
                order_data['bundle_id'],
                inner_orders,
                order_data['can_revert'],
                order_data['gas_used']
            )

        raise ValueError(f"Unknown order type for deserialization: {order_type}")


class TxOrder(Order):
    """
    Represents a single transaction order, holding a fully parsed py-evm transaction object.
    """
    def __init__(self, timestamp_ms: int, raw_tx: bytes, tx_object: SignedTransactionAPI):
        self.timestamp_ms = timestamp_ms
        self.raw_tx = raw_tx
        self.tx_obj: SignedTransactionAPI = tx_object

    def id(self) -> OrderId:
        return OrderId.from_tx(self.tx_obj.hash)

    def order_type(self) -> OrderType:
        return OrderType.TX

    def can_execute_with_block_base_fee(self, block_base_fee: int) -> bool:
        """Checks if the transaction's fee cap is sufficient for the block's base fee."""
        return self.tx_obj.max_fee_per_gas >= block_base_fee

    def nonces(self) -> List[TxNonce]:
        """Returns the nonce this transaction depends on."""
        checksum_address = Web3.to_checksum_address(self.tx_obj.sender.hex())
        return [TxNonce(address=checksum_address, nonce=self.tx_obj.nonce, optional=False)]

    def transactions(self) -> List[Dict[str, Any]]:
        """Returns a list containing this transaction's summary."""
        tx_hash = self.id().value
        return [{
            'hash': tx_hash,
            'from': self.tx_obj.sender.hex(),
            'to': self.tx_obj.to.hex() if self.tx_obj.to else None,
            'nonce': self.tx_obj.nonce,
            'raw_tx': f"0x{self.raw_tx.hex()}"
        }]

    def get_vm_transaction(self) -> SignedTransactionAPI:
        """
        Returns the pre-parsed transaction object ready for EVM simulation.
        """
        return self.tx_obj

    def order_data(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "raw_tx": self.raw_tx,
            "tx_hash": self.id().value,
        }

    @classmethod
    def from_raw(cls, timestamp_ms: int, raw_tx: bytes) -> 'TxOrder':
        """
        Decodes a raw transaction using RLP sedes and wraps it in the final
        py-evm object (PragueLegacy/PragueTyped) ready for simulation.
        """
        if not raw_tx:
            raise ValueError("Empty transaction data")

        try:
            first_byte = raw_tx[0]
            final_tx_obj: SignedTransactionAPI = None

            if first_byte <= 0x7f:  # Typed Transaction
                tx_type = first_byte
                payload = raw_tx[1:]
                inner_tx = None

                if tx_type == 0x01:
                    inner_tx = rlp.decode(payload, sedes=AccessListTransaction)
                elif tx_type == 0x02:
                    inner_tx = rlp.decode(payload, sedes=DynamicFeeTransaction)
                elif tx_type == 0x03:
                    wrapper = rlp.decode(payload)
                    inner_payload = rlp.encode(wrapper[0])
                    inner_tx = rlp.decode(inner_payload, sedes=BlobTransaction)
                elif tx_type == 0x04:
                    inner_tx = rlp.decode(payload, sedes=SetCodeTransaction)
                else:
                    raise ValueError(f"Unsupported transaction type: {tx_type}")

                # Wrap the inner transaction to create the final, usable object
                final_tx_obj = PragueTypedTransaction(tx_type, inner_tx)
            
            else:  # Legacy Transaction
                # Legacy transactions are decoded directly into their final form
                final_tx_obj = rlp.decode(raw_tx, sedes=PragueLegacyTransaction)

            return cls(timestamp_ms, raw_tx, final_tx_obj)

        except (ValueError, TypeError, IndexError, DecodingError) as e:
            raise ValueError(f"Failed to parse raw tx: {e}")


def _can_execute_list_txs(
    list_txs: List[Tuple['Order', bool]],
    block_base_fee: int
) -> bool:
    """
    Checks that at least one tx can execute and all mandatory txs can.
    """
    any_can_execute = False
    for order, is_optional in list_txs:
        can_execute = order.can_execute_with_block_base_fee(block_base_fee)
        if can_execute:
            any_can_execute = True
        elif not is_optional:
            # Mandatory tx cannot execute
            return False
    return any_can_execute


@dataclass
class BundleOrder(Order):
    bundle_uuid: uuid.UUID
    # list of (Order, optional flag)
    child_orders: List[Tuple[Order, bool]]

    def id(self) -> OrderId:
        return OrderId.from_bundle(self.bundle_uuid)

    def order_type(self) -> OrderType:
        return OrderType.BUNDLE

    def order_data(self) -> Dict[str, Any]:
        return {"bundle_uuid": str(self.bundle_uuid)}

    def can_execute_with_block_base_fee(self, block_base_fee: int) -> bool:
        return _can_execute_list_txs(self.child_orders, block_base_fee)

    def nonces(self) -> List[TxNonce]:
        """Return list of nonces this order depends on."""
        nonces = []
        for order, optional in self.child_orders:
            for nonce in order.nonces():
                nonces.append(TxNonce(address=nonce.address, nonce=nonce.nonce, optional=optional))
        return nonces

    def transactions(self) -> List[Dict[str, Any]]:
        """Return list of transactions in this bundle."""
        transactions = []
        for order, optional in self.child_orders:
            transactions.extend(order.transactions())
        return transactions

@dataclass
class ShareBundleOrder(Order):
    share_hash: bytes
    child_orders: List[Tuple[Order, bool]]

    def id(self) -> OrderId:
        return OrderId.from_sharebundle(self.share_hash)

    def order_type(self) -> OrderType:
        return OrderType.SHAREBUNDLE

    def order_data(self) -> Dict[str, Any]:
        return {"share_hash": self.id().value}

    def can_execute_with_block_base_fee(self, block_base_fee: int) -> bool:
        return _can_execute_list_txs(self.child_orders, block_base_fee)

    def nonces(self) -> List[TxNonce]:
        """Return list of nonces this order depends on."""
        nonces = []
        for order, optional in self.child_orders:
            for nonce in order.nonces():
                nonces.append(TxNonce(address=nonce.address, nonce=nonce.nonce, optional=optional))
        return nonces

    def transactions(self) -> List[Dict[str, Any]]:
        """Return list of transactions in this sharebundle."""
        transactions = []
        for order, optional in self.child_orders:
            transactions.extend(order.transactions())
        return transactions

def make_tx_orders(raw_records: List[Dict[str, Any]]) -> List[TxOrder]:
    orders = []
    for r in raw_records:
        try:
            order = TxOrder.from_raw(r['timestamp_ms'], r['raw_tx'])
            orders.append(order)
        except ValueError as e:
            # Extract the original error message to match Rust format exactly
            error_msg = str(e)
            if error_msg.startswith("Failed to parse raw tx: "):
                original_error = error_msg[len("Failed to parse raw tx: "):]
                logger.error(f"Failed to parse raw tx: {original_error}")
            else:
                logger.error(f"Failed to parse raw tx: {error_msg}")
        except Exception as e:
            logger.error(f"Failed to parse raw tx: {e}")
    return orders

def fetch_transactions(
    parquet_files: List[str],
    from_ts_ms: int,
    to_ts_ms: int
) -> List[TxOrder]:
    orders = []
    for path in parquet_files:
        df = pd.read_parquet(path, columns=["timestamp", "rawTx"])
        df = df.rename(columns={"timestamp": "timestamp_dt", "rawTx": "raw_tx"})
        df["timestamp_ms"] = (
            df["timestamp_dt"].values.astype("datetime64[ms]").astype("int64")
        )
        df = df[
            (df["timestamp_ms"] > from_ts_ms) &
            (df["timestamp_ms"] < to_ts_ms)
        ]

        for ts, raw in zip(df["timestamp_ms"], df["raw_tx"]):
            try:
                tx = TxOrder.from_raw(int(ts), raw)
                orders.append(tx)
            except ValueError as e:
                # Extract the original error message to match Rust format exactly
                error_msg = str(e)
                if error_msg.startswith("Failed to parse raw tx: "):
                    original_error = error_msg[len("Failed to parse raw tx: "):]
                    logger.error(f"Failed to parse raw tx: {original_error}")
                else:
                    logger.error(f"Failed to parse raw tx: {error_msg}")
                continue
            except Exception as e:
                logger.error(f"Failed to parse raw tx: {e}")
                continue

    orders.sort(key=lambda o: o.timestamp_ms)
    return orders

def filter_orders_by_base_fee(
    block_base_fee: int,
    orders: List[Order]
) -> List[Order]:
    return [o for o in orders if o.can_execute_with_block_base_fee(block_base_fee)]

def filter_orders_by_nonces(
    provider: RootProvider,
    orders: List[Order],
    block_number: int,
    concurrency_limit: int,
) -> List[Order]:
    """
    Filters out orders whose non-optional sub-txns have already been mined.
    
    This mirrors the Rust implementation: for each order, check its nonces against
    the on-chain state at parent_block. If any non-optional nonce is too low 
    (onchain_nonce > tx_nonce), drop the order. If all nonces failed, also drop.
    
    Returns the filtered list.
    """
    parent_block = block_number - 1
    
    # Collect all unique addresses to fetch nonces for
    unique_addresses = set()
    for order in orders:
        for nonce in order.nonces():
            unique_addresses.add(nonce.address)

    # Fetch nonces concurrently
    nonce_cache: Dict[str, int] = {}
    lock = Lock()

    def fetch_and_cache_nonce(address: str):
        try:
            nonce = provider.w3.eth.get_transaction_count(address, parent_block)
            with lock:
                nonce_cache[address] = nonce
        except Exception as e:
            logger.warning(f"Could not fetch nonce for {address}@{parent_block}: {e}")
            # If fetching fails, we can't validate, so we store -1 to indicate failure
            with lock:
                nonce_cache[address] = -1

    with ThreadPoolExecutor(max_workers=concurrency_limit) as executor:
        executor.map(fetch_and_cache_nonce, unique_addresses)

    # Filter orders using cache
    kept: List[Order] = []
    for order in orders:
        order_nonces = order.nonces()
        all_nonces_failed = True
        should_drop = False

        for nonce in order_nonces:
            onchain_nonce = nonce_cache.get(nonce.address)

            if onchain_nonce is None or onchain_nonce == -1:
                should_drop = True
                break

            # Check if this nonce is too low
            if onchain_nonce > nonce.nonce and not nonce.optional:
                logger.debug(
                    f"Order nonce too low, order: {order.id()}, nonce: {nonce.nonce}, onchain tx count: {onchain_nonce}"
                )
                should_drop = True
                break
            
            if onchain_nonce <= nonce.nonce:
                # This nonce is still valid, so not all nonces have failed
                all_nonces_failed = False

        if should_drop:
            continue

        if all_nonces_failed:
            logger.debug(f"All nonces failed, order: {order.id()}")
            continue

        logger.debug(f"Order nonce ok, order: {order.id()}")
        kept.append(order)

    return kept
