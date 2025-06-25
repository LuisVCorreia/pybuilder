from web3 import Web3, HTTPProvider
from typing import Any, Dict, Optional

class RootProvider:
    """
    RootProvider fetches block data from an Ethereum node using web3.py.
    """
    def __init__(self, rpc_url: str):
        self.w3 = Web3(HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to Ethereum node at {rpc_url}")

    def get_block(self, block_number: int, full_transactions: bool = True) -> Dict[str, Any]:
        """
        Fetch a block by number.
        Args:
            block_number: The block number to fetch.
            full_transactions: If True, include full tx objects; else just hashes.
        Returns:
            Block data as a dictionary (web3.py format).
        Raises:
            ValueError if the block is not found.
        """
        try:
            block = self.w3.eth.get_block(block_number, full_transactions=full_transactions)
            return dict(block)
        except Exception as e:
            raise ValueError(f"Failed to fetch block {block_number}: {e}")
