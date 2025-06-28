import enum
import uuid
from dataclasses import dataclass
from typing import Any, List, Union, Dict, Tuple
from eth_utils import keccak
import rlp
from rlp.exceptions import DecodingError
import pandas as pd
import logging
from ..fetch.root_provider import RootProvider
from eth_account import Account

logger = logging.getLogger(__name__)


def safe_int_from_field(field) -> int:
    """Safely convert RLP field to int, handling bytes and int."""
    if isinstance(field, bytes):
        if not field:
            return 0
        return int.from_bytes(field, 'big')
    
    if isinstance(field, int):
        return field

    # If we get here, something is wrong. Log it and return 0 to be safe.
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
    Unique identifier for an Order; mirrors Rust's OrderId enum.
    Formats:
      - tx:0x<hash>
      - bundle:<uuid>
      - sbundle:0x<hash>
    """
    type: OrderType
    value: str

    def __str__(self) -> str:
        return f"{self.type.value}:{self.value}"

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

class TxOrder(Order):
    def __init__(self, timestamp_ms: int, raw_tx: bytes, max_fee_per_gas: int, nonce: int, sender: str, canonical_tx: bytes = None):
        self.timestamp_ms = timestamp_ms
        self.raw_tx = raw_tx
        self.max_fee_per_gas = max_fee_per_gas
        self.nonce = nonce
        self.sender = sender
        # The canonical transaction is used for hashing. For most tx types, it's
        # the same as the raw_tx. For EIP-4844 blob txs, it's a specific
        # representation without the blob sidecar.
        self.canonical_tx = canonical_tx if canonical_tx is not None else raw_tx

    def id(self) -> OrderId:
        txhash = keccak(self.canonical_tx)
        return OrderId.from_tx(txhash)

    def order_type(self) -> OrderType:
        return OrderType.TX

    def order_data(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "raw_tx": self.raw_tx,
            "tx_hash": self.id().value,
        }

    @classmethod
    def from_raw(cls, timestamp_ms: int, raw_tx: bytes) -> 'TxOrder':
        """
        Decode the raw signed RLP once to extract:
         - max_fee_per_gas
         - nonce
         - sender (via signature recovery)
        
        This implementation handles different transaction types, including
        EIP-4844 blob transactions. For blob transactions, which are encoded
        in a network-specific format (with sidecar), it constructs the
        canonical transaction representation to ensure compatibility with
        sender recovery logic in `eth-account` and for correct hash calculation.
        """
        try:
            if not raw_tx:
                raise ValueError("Empty transaction data")

            first_byte = raw_tx[0]
            tx_type = None
            sender = None
            canonical_tx = raw_tx  # Default to raw_tx, override for blob txs

            if first_byte <= 0x7f:  # Typed transaction
                tx_type = first_byte
                payload = raw_tx[1:]
                if not payload:
                    raise ValueError(f"Empty payload for typed transaction {tx_type}")
                
                decoded_payload = rlp.decode(payload)
                if not isinstance(decoded_payload, list):
                    raise ValueError("Transaction payload must be a list")

                if tx_type == 0x03:  # EIP-4844 Blob Transaction
                    if len(decoded_payload) < 1:
                        raise ValueError("Invalid EIP-4844 payload structure")
                    
                    tx_payload_body = decoded_payload[0]
                    if not isinstance(tx_payload_body, list):
                        raise ValueError("EIP-4844 transaction body must be a list of fields")

                    fields = tx_payload_body
                    if len(fields) < 9: # chainId, nonce, maxPriorityFeePerGas, maxFeePerGas, gas, to, value, data, accessList, ...
                        raise ValueError(f"EIP-4844 tx body needs at least 9 fields, got {len(fields)}")
                    
                    nonce = safe_int_from_field(fields[1])
                    max_fee = safe_int_from_field(fields[3])

                    # For blob txs, the hash and signature are based on the canonical
                    # transaction form, not the full network RLP with sidecar.
                    try:
                        canonical_tx_encoded = rlp.encode(tx_payload_body)
                        canonical_tx = raw_tx[0:1] + canonical_tx_encoded
                        sender = Account.recover_transaction(canonical_tx)
                    except Exception as e:
                        raise ValueError(f"Failed to recover sender from canonical blob tx: {e}")

                else: # Other typed transactions (EIP-2930, EIP-1559, etc.)
                    fields = decoded_payload
                    if tx_type == 0x01:  # EIP-2930
                        if len(fields) < 8:
                            raise ValueError(f"EIP-2930 tx needs at least 8 fields, got {len(fields)}")
                        nonce = safe_int_from_field(fields[1])
                        max_fee = safe_int_from_field(fields[2]) # gasPrice
                    elif tx_type == 0x02:  # EIP-1559
                        if len(fields) < 9:
                            raise ValueError(f"EIP-1559 tx needs at least 9 fields, got {len(fields)}")
                        nonce = safe_int_from_field(fields[1])
                        max_fee = safe_int_from_field(fields[3]) # maxFeePerGas
                    elif tx_type == 0x04: # EIP-7702
                        if len(fields) < 9:
                            raise ValueError(f"EIP-7702 tx needs at least 9 fields, got {len(fields)}")
                        nonce = safe_int_from_field(fields[1])
                        max_fee = safe_int_from_field(fields[3]) # maxFeePerGas
                    else: # Other or unknown typed tx
                        if len(fields) < 3:
                             raise ValueError(f"Unsupported typed tx {tx_type} with {len(fields)} fields")
                        nonce = safe_int_from_field(fields[1])
                        max_fee = 0 # Unknown, default to 0

                    
                    # For non-blob typed txs, recover directly from the raw transaction
                    sender = Account.recover_transaction(raw_tx)

            else:  # Legacy transaction
                fields = rlp.decode(raw_tx)
                if not isinstance(fields, list) or len(fields) < 6:
                    raise ValueError(f"Legacy transaction needs at least 6 fields, got {len(fields)}")
                nonce = safe_int_from_field(fields[0])
                max_fee = safe_int_from_field(fields[1]) # gasPrice
                
                # For legacy txs, recover directly from the raw transaction
                sender = Account.recover_transaction(raw_tx)

            if not sender or sender == "0x0000000000000000000000000000000000000000":
                raise ValueError("Failed to recover valid sender address")

            return cls(timestamp_ms, raw_tx, max_fee, nonce, sender, canonical_tx)
    
        except (ValueError, TypeError, IndexError, DecodingError) as e:
            # Catch specific, expected errors and wrap them.
            raise ValueError(f"Failed to parse raw tx: {e}")
        except Exception as e:
            # Catch any other unexpected errors.
            # This will also catch errors from Account.recover_transaction
            raise ValueError(f"Failed to parse raw tx: {e}")
    
    def can_execute_with_block_base_fee(self, block_base_fee: int) -> bool:
        return self.max_fee_per_gas >= block_base_fee

    def nonces(self) -> List[TxNonce]:
        """Return list of nonces this order depends on."""
        return [TxNonce(address=self.sender, nonce=self.nonce, optional=False)]

def _can_execute_list_txs(
    list_txs: List[Tuple['Order', bool]],
    block_base_fee: int
) -> bool:
    """
    Checks that at least one tx can execute and all mandatory txs can.
    """
    executable_count = 0
    for order, optional in list_txs:
        if order.can_execute_with_block_base_fee(block_base_fee):
            executable_count += 1
        elif not optional:
            return False
    return executable_count > 0


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
    block_number: int
) -> List[Order]:
    """
    Filters out orders whose non-optional sub-txns have already been mined.
    
    This mirrors the Rust implementation: for each order, check its nonces against
    the on-chain state at parent_block. If any non-optional nonce is too low 
    (onchain_nonce > tx_nonce), drop the order. If all nonces failed, also drop.
    
    Returns the filtered list.
    """
    parent_block = block_number - 1
    nonce_cache: Dict[str, int] = {}
    kept: List[Order] = []

    for order in orders:
        order_nonces = order.nonces()
        all_nonces_failed = True
        should_drop = False

        for nonce in order_nonces:
            # Check cache first
            onchain_nonce = nonce_cache.get(nonce.address)
            
            if onchain_nonce is None:
                try:
                    onchain_nonce = provider.w3.eth.get_transaction_count(nonce.address, parent_block)
                    nonce_cache[nonce.address] = onchain_nonce
                except Exception as e:
                    logger.debug(f"Could not fetch nonce for {nonce.address}@{parent_block}: {e}")
                    # If we can't get the nonce, be conservative and drop the order
                    should_drop = True
                    break

            # Check if this nonce is too low (already mined)
            if onchain_nonce > nonce.nonce and not nonce.optional:
                logger.debug(
                    f"Order nonce too low, order: {order.id()}, nonce: {nonce.nonce}, onchain tx count: {onchain_nonce}"
                )
                should_drop = True
                break
            elif onchain_nonce <= nonce.nonce:
                # This nonce is still valid
                all_nonces_failed = False

        if should_drop:
            continue

        if all_nonces_failed:
            logger.debug(f"All nonces failed, order: {order.id()}")
            continue

        logger.debug(f"Order nonce ok, order: {order.id()}")
        kept.append(order)

    return kept
