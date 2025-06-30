import sqlite3
import json
import zlib
import struct
import logging
from decimal import Decimal
from backtest.common.order import Order
from .block_data import BlockData, BuiltBlockData, OrdersWithTimestamp

logger = logging.getLogger(__name__)

class HistoricalDataStorage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        
    def create_tables(self):
        """Create all required tables"""
        c = self.conn.cursor()

        # Create blocks table
        c.execute("""
        CREATE TABLE IF NOT EXISTS blocks (
            block_number INTEGER NOT NULL,
            block_hash TEXT NOT NULL,
            fee_recipient TEXT NOT NULL,
            bid_value TEXT NOT NULL
        )
        """)       
        
        # Create orders table
        c.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            block_number INTEGER NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            order_type TEXT NOT NULL,
            order_id TEXT,
            order_data BLOB NOT NULL
        )
        """)

        # Create blocks_data table
        c.execute("""
        CREATE TABLE IF NOT EXISTS blocks_data (
            block_number INTEGER NOT NULL,
            winning_bid_trace BLOB NOT NULL,
            onchain_block BLOB NOT NULL
        )
        """)
        
        # Create built_block_data table
        c.execute("""
        CREATE TABLE IF NOT EXISTS built_block_data (
            block_number INTEGER NOT NULL,
            orders_closed_at_ts_ms INTEGER NOT NULL,
            sealed_at_ts_ms INTEGER NOT NULL,
            profit TEXT NOT NULL
        )
        """)
        
        # Create built_block_included_orders table
        c.execute("""
        CREATE TABLE IF NOT EXISTS built_block_included_orders (
            block_number INTEGER NOT NULL,
            order_id TEXT NOT NULL
        )
        """)
        
        # Create indexes for performance
        c.execute("CREATE INDEX IF NOT EXISTS orders_block_number_idx ON orders (block_number)")
        c.execute("CREATE INDEX IF NOT EXISTS blocks_block_number_idx ON blocks (block_number)")
        
        self.conn.commit()

    def write_block_data(self, block_data: 'BlockData'):
        """Write complete block data"""
        c = self.conn.cursor()
        
        # Delete existing data for this block
        c.execute("DELETE FROM blocks WHERE block_number = ?", (block_data.block_number,))
        c.execute("DELETE FROM built_block_included_orders WHERE block_number = ?", (block_data.block_number,))
        c.execute("DELETE FROM built_block_data WHERE block_number = ?", (block_data.block_number,))
        c.execute("DELETE FROM orders WHERE block_number = ?", (block_data.block_number,))
        c.execute("DELETE FROM blocks_data WHERE block_number = ?", (block_data.block_number,))
        
        # Insert block data
        block_hash = block_data.winning_bid_trace.get('block_hash', '')
        fee_recipient = block_data.winning_bid_trace.get('proposer_fee_recipient', '')
        bid_value = block_data.winning_bid_trace.get('value', '0')
        c.execute("""
        INSERT INTO blocks (block_number, block_hash, fee_recipient, bid_value)
        VALUES (?, ?, ?, ?)
        """, (block_data.block_number, block_hash, fee_recipient, str(bid_value)))

        # Insert orders
        for order_with_ts in block_data.available_orders:
            order = order_with_ts.order
            order_id = str(order.id())
            order_type = order.order_type().value
            order_data = order.order_data()
            
            serialized_data = json.dumps(
                order_data,
                default=lambda o: o.hex() if isinstance(o, (bytes, bytearray)) else str(o)
            ).encode('utf-8')
            
            compressed_data = compress_prepend_size(serialized_data)
            
            c.execute("""
            INSERT INTO orders (block_number, timestamp_ms, order_type, order_id, order_data)
            VALUES (?, ?, ?, ?, ?)
            """, (block_data.block_number, order_with_ts.timestamp_ms, order_type, order_id, compressed_data))

        # Insert blocks_data
        winning_bid_trace_json = compress_prepend_size(json.dumps(block_data.winning_bid_trace, default=str).encode('utf-8'))
        onchain_block_json = compress_prepend_size(json.dumps(block_data.onchain_block, default=str).encode('utf-8'))

        c.execute("""
        INSERT INTO blocks_data (block_number, winning_bid_trace, onchain_block)
        VALUES (?, ?, ?)
        """, (block_data.block_number, winning_bid_trace_json, onchain_block_json))
        
        # Insert built_block_data if provided
        if block_data.built_block_data:
            built_data = block_data.built_block_data
            orders_closed_at_ts_ms = int(built_data.orders_closed_at.timestamp() * 1000)
            sealed_at_ts_ms = int(built_data.sealed_at.timestamp() * 1000)
            profit_str = str(built_data.profit)
            
            c.execute("""
            INSERT INTO built_block_data (block_number, orders_closed_at_ts_ms, sealed_at_ts_ms, profit)
            VALUES (?, ?, ?, ?)
            """, (block_data.block_number, orders_closed_at_ts_ms, sealed_at_ts_ms, profit_str))
            
            # Insert included orders
            for order_id in built_data.included_orders:
                c.execute("""
                INSERT INTO built_block_included_orders (block_number, order_id)
                VALUES (?, ?)
                """, (block_data.block_number, order_id))
        
        self.conn.commit()

    def read_block_data(self, block_number: int) -> 'BlockData':
        """Read block data by block number"""
        c = self.conn.cursor()
        
        # Read block info
        c.execute("SELECT * FROM blocks WHERE block_number = ?", (block_number,))
        block_row = c.fetchone()
        if not block_row:
            raise ValueError(f"No data found for block {block_number}")
        
        block_data = {
            'block_number': block_row[0],
            'block_hash': block_row[1],
            'fee_recipient': block_row[2],
            'bid_value': Decimal(block_row[3])
        }
        
        # Read orders
        c.execute("SELECT * FROM orders WHERE block_number = ?", (block_number,))
        orders = []
        for row in c.fetchall():
            order_type = row[2]
            timestamp_ms = row[1]
            order_data = decompress_size_prepended(row[4])
            orders.append(OrdersWithTimestamp(timestamp_ms, Order.from_serialized(order_type, order_data)))

        # Read blocks_data
        c.execute("SELECT * FROM blocks_data WHERE block_number = ?", (block_number,))
        blocks_data_row = c.fetchone()
        if not blocks_data_row:
            raise ValueError(f"No blocks data found for block {block_number}")
        
        winning_bid_trace_json = decompress_size_prepended(blocks_data_row[1])
        winning_bid_trace = json.loads(winning_bid_trace_json.decode('utf-8'))

        onchain_block_json = decompress_size_prepended(blocks_data_row[2])
        onchain_block = json.loads(onchain_block_json.decode('utf-8'))
        
        # Read built_block_data
        built_block_data = None
        c.execute("SELECT * FROM built_block_data WHERE block_number = ?", (block_number,))
        built_row = c.fetchone()
        
        if built_row:
            orders_closed_at_ts_ms = built_row[1]
            sealed_at_ts_ms = built_row[2]
            profit_str = built_row[3]
            
            from datetime import datetime
            orders_closed_at = datetime.fromtimestamp(orders_closed_at_ts_ms / 1000)
            sealed_at = datetime.fromtimestamp(sealed_at_ts_ms / 1000)
            
            built_block_data = BuiltBlockData(
                included_orders=[],
                orders_closed_at=orders_closed_at,
                sealed_at=sealed_at,
                profit=Decimal(profit_str)
            )
            
            # Read included orders
            c.execute("SELECT order_id FROM built_block_included_orders WHERE block_number = ?", (block_number,))
            for inc_order in c.fetchall():
                built_block_data.included_orders.append(inc_order[0])
        else:
            built_block_data = None
        # Create BlockData instance
        block_data_instance = BlockData(
            block_number=block_data['block_number'],
            winning_bid_trace=winning_bid_trace,
            onchain_block=onchain_block,
            available_orders=orders,
            built_block_data=built_block_data
        )
        return block_data_instance
    
    def close(self):
        """Close the database connection"""
        self.conn.close()


# Compression utilities
def compress_prepend_size(input_bytes: bytes) -> bytes:
    """Compress all bytes of input_bytes, prefixing the uncompressed length as a little-endian u32."""
    compressed = zlib.compress(input_bytes)
    return struct.pack('<I', len(input_bytes)) + compressed


def decompress_size_prepended(data: bytes) -> bytes:
    """Given bytes prefixed by a little-endian u32 length, decompress the rest."""
    size, = struct.unpack('<I', data[:4])
    decompressed = zlib.decompress(data[4:])
    if len(decompressed) != size:
        raise ValueError(f"Decompressed size {len(decompressed)} does not match prefix {size}")
    return decompressed
