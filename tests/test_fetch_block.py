import os
import tempfile
import pytest
from backtest_fetch.fetch_block import fetch_and_store_block

def test_fetch_and_store_block():
    provider_url = os.environ.get("TEST_ETH_PROVIDER")
    if not provider_url:
        pytest.skip("TEST_ETH_PROVIDER not set")
    mempool_data_dir = tempfile.mkdtemp()
    db_path = os.path.join(tempfile.gettempdir(), "test_backtest.sqlite")
    block_number = 18000000  # Use a known block on mainnet or testnet
    try:
        fetch_and_store_block(
            block_number=block_number,
            provider_url=provider_url,
            mempool_data_dir=mempool_data_dir,
            sqlite_db_path=db_path,
            window_before_sec=60,
            window_after_sec=5,
        )
        assert os.path.exists(db_path)
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
