import sqlite3
from typing import Optional, Dict, List
from .order import Order
import json
import zlib
import struct

def init_db(db_path: str) -> sqlite3.Connection:
    """
    Initialize SQLite DB, create tables if missing.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Create blocks table with block_number as primary key
    c.execute("""
    CREATE TABLE IF NOT EXISTS blocks (
        block_number INTEGER PRIMARY KEY,
        block_hash TEXT NOT NULL,
        fee_recipient TEXT NOT NULL,
        bid_value TEXT NOT NULL
    );
    """)
    # Create orders table
    c.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        block_number INTEGER NOT NULL,
        timestamp_ms INTEGER NOT NULL,
        order_type TEXT NOT NULL,
        order_id TEXT,
        order_data BLOB NOT NULL,
        FOREIGN KEY(block_number) REFERENCES blocks(block_number)
    );
    """)
    # Index on block_number for fast lookup
    c.execute("CREATE INDEX IF NOT EXISTS orders_block_number_idx ON orders (block_number);")
    c.execute("CREATE INDEX IF NOT EXISTS blocks_block_number_idx ON blocks (block_number);")
    conn.commit()
    return conn


def insert_block(
    conn: sqlite3.Connection,
    bid_trace: Dict
):
    """
    Insert a block record, replacing any existing one (and its orders).
    """
    c = conn.cursor()
    # Delete existing orders and block if present
    c.execute("DELETE FROM orders WHERE block_number = ?", (bid_trace["block_number"],))
    c.execute("DELETE FROM blocks WHERE block_number = ?", (bid_trace["block_number"],))
    # Insert new block record
    c.execute(
        "INSERT INTO blocks (block_number, block_hash, fee_recipient, bid_value) VALUES (?, ?, ?, ?)",
        (bid_trace["block_number"], bid_trace["block_hash"], bid_trace["proposer_fee_recipient"], str(bid_trace["value"]))
    )
    conn.commit()


def insert_order(
    conn: sqlite3.Connection,
    block_number: int,
    timestamp_ms: int,
    order_type: str,
    order_id: Optional[str],
    order_data: bytes
):
    """
    Insert a single order for the given block.
    """
    c = conn.cursor()
    c.execute(
        "INSERT INTO orders (block_number, timestamp_ms, order_type, order_id, order_data) VALUES (?, ?, ?, ?, ?)",
        (block_number, timestamp_ms, order_type, order_id, order_data)
    )
    conn.commit()


def compress_prepend_size(input_bytes: bytes) -> bytes:
    """
    Compress all bytes of `input_bytes`, prefixing the uncompressed length as a little-endian u32.
    Equivalent to Rust's compress_prepend_size.
    """
    compressed = zlib.compress(input_bytes)
    return struct.pack('<I', len(input_bytes)) + compressed


def decompress_size_prepended(data: bytes) -> bytes:
    """
    Given bytes prefixed by a little-endian u32 length, decompress the rest.
    Equivalent to Rust's decompress_size_prepended.
    """
    size, = struct.unpack('<I', data[:4])
    decompressed = zlib.decompress(data[4:])
    if len(decompressed) != size:
        raise ValueError(f"Decompressed size {len(decompressed)} does not match prefix {size}")
    return decompressed

def write_block_data(
    conn: sqlite3.Connection,
    block_number: int,
    bid_trace: Dict,
    orders: List[Order]
):
    """
    Write a block and its orders to the database.

    Uses insert_block/insert_order under the hood for clarity.
    Deletes any existing data for the block first to prevent duplicates.
    """
    # 1) Insert or replace block (also clears old orders)
    insert_block(conn, bid_trace)

    # 2) Insert each order
    for order in orders:
        oid = str(order.id())
        otype = order.order_type().value
        odata = order.order_data()
        serialized_data = json.dumps(
            odata,
            default=lambda o: o.hex() if isinstance(o, (bytes, bytearray)) else str(o)
        ).encode('utf-8')

        # Compress the JSON payload
        compressed_data = compress_prepend_size(serialized_data)

        insert_order(
            conn,
            block_number,
            int(odata.get('timestamp_ms', 0)),
            otype,
            oid,
            compressed_data
        )
