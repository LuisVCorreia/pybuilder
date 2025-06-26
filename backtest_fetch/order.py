import enum
import uuid
from dataclasses import dataclass
from typing import Any, List, Union, Dict, Tuple
from eth_utils import keccak
import rlp
from rlp.exceptions import DecodingError
import pandas as pd
import logging

logger = logging.getLogger(__name__)


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

@dataclass
class TxOrder(Order):
    timestamp_ms: int
    raw_tx: bytes
    max_fee_per_gas: int

    def id(self) -> OrderId:
        txhash = keccak(self.raw_tx)
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
        first = raw_tx[0]
        try:
            # strip type‐byte for typed txs
            payload = raw_tx[1:] if first in (0x01, 0x02, 0x07) else raw_tx
            fields = rlp.decode(payload, strict=False)
            if first == 0x01:           # EIP-2930
                fee_bytes = fields[2]
            elif first in (0x02,0x07):  # EIP-1559 & 4844
                fee_bytes = fields[3]
            else:                       # legacy
                fee_bytes = fields[1]

            max_fee = int.from_bytes(fee_bytes, "big")
            return cls(timestamp_ms, raw_tx, max_fee)

        except (DecodingError, IndexError) as e:
            logger.error("Failed to decode raw tx at %d: %s", timestamp_ms, e)
            raise
    
    def can_execute_with_block_base_fee(self, block_base_fee: int) -> bool:
        try:
            return self.max_fee_per_gas >= block_base_fee
        except Exception:
            # any error => treat as non-executable (drop it)
            return False

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

def make_tx_orders(raw_records: List[Dict[str, Any]]) -> List[TxOrder]:
    return [TxOrder(r['timestamp_ms'], r['raw_tx']) for r in raw_records]

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
            except Exception:
                # Already logged inside from_raw
                continue
            orders.append(tx)

    orders.sort(key=lambda o: o.timestamp_ms)
    return orders

def filter_orders_by_base_fee(
    block_base_fee: int,
    orders: List[Order]
) -> List[Order]:
    return [o for o in orders if o.can_execute_with_block_base_fee(block_base_fee)]
