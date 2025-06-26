import enum
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional, Union, Dict
from eth_utils import keccak
from web3 import Web3

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
        except Exception as e:
            raise ValueError(f"Invalid OrderId string: {text}")
        if otype == OrderType.BUNDLE:
            u = uuid.UUID(v)
            return cls(type=otype, value=str(u))
        else:
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
        """Return the type of this order."""
        raise NotImplementedError

    def order_data(self) -> Dict[str, Any]:
        """Return a dict of the order's raw data (e.g. timestamp, raw tx, etc)."""
        raise NotImplementedError

@dataclass
class TxOrder(Order):
    timestamp_ms: int   # Timestamp of when the tx entered the mempool
    raw_tx: bytes       # Signed RLP

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

@dataclass
class BundleOrder(Order):
    bundle_uuid: uuid.UUID

    def id(self) -> OrderId:
        return OrderId.from_bundle(self.bundle_uuid)

    def order_type(self) -> OrderType:
        return OrderType.BUNDLE

    def order_data(self) -> Dict[str, Any]:
        return {"bundle_uuid": str(self.bundle_uuid)}

@dataclass
class ShareBundleOrder(Order):
    share_hash: bytes

    def id(self) -> OrderId:
        return OrderId.from_sharebundle(self.share_hash)

    def order_type(self) -> OrderType:
        return OrderType.SHAREBUNDLE

    def order_data(self) -> Dict[str, Any]:
        return {"share_hash": self.id().value}

def make_tx_orders(raw_records: List[Dict[str, Any]]) -> List[TxOrder]:
    """
    Given a list of dicts with keys 'timestamp_ms' and 'raw_tx', produce TxOrder objects.
    """
    return [TxOrder(r['timestamp_ms'], r['raw_tx']) for r in raw_records]

import pandas as pd
from typing import List

def filter_transactions(
    parquet_files: List[str],
    from_ts_ms: int,
    to_ts_ms: int
) -> List[TxOrder]:
    parts = []
    for path in parquet_files:
        df = pd.read_parquet(path, columns=["timestamp", "rawTx"])
        df = df.rename(columns={"timestamp": "timestamp_dt", "rawTx": "raw_tx"})
        df["timestamp_ms"] = (
            df["timestamp_dt"].values.astype("datetime64[ms]").astype("int64")
        )
        mask = (df["timestamp_ms"] > from_ts_ms) & (df["timestamp_ms"] < to_ts_ms)
        sliced = df.loc[mask, ["timestamp_ms", "raw_tx"]]
        if sliced.empty:
            continue
        parts.append(sliced.to_dict(orient="records"))

    if not parts:
        return []

    all_records = [rec for part in parts for rec in part]
    all_records.sort(key=lambda r: r["timestamp_ms"])
    return make_tx_orders(all_records)
