from typing import Optional, Dict
from .state_provider import StateProvider, StateProviderFactory, AccountInfo, SimulationContext


class MockStateProvider(StateProvider):
    """Mock implementation with hardcoded state data"""
    
    def __init__(self, block_number: int = 20114954):
        self.block_number = block_number
        
        # Cache for dynamic accounts
        self._dynamic_accounts: Dict[str, AccountInfo] = {}
        
        # Mock storage data
        self.storage = {}
        
        # Mock contract code
        self.contract_code = {
            "0xa0b86a33e6417c9dce0fb0503df7b2e4c0cd3c4b": "0x608060405234801561001057600080fd5b50600436106100415760003560e01c80633fb5c1cb1461004657806341c0e1b514610062578063cfae321714610088575b600080fd5b61006060048036038101906100599190610123565b6100a6565b005b61006a6100b0565b604051610081929190610160565b60405180910390f35b6100906100d9565b60405161009d9190610190565b60405180910390f35b8060008190555050565b60008060009054906101000a900473ffffffffffffffffffffffffffffffffffffffff16600054915091509091565b606060405180602001604052806000815250905090565b600080fd5b6000819050919050565b61010b816100f8565b811461011657600080fd5b50565b60008135905061012881610102565b92915050565b60006020828403121561013957610138610123565b5b600061014784828501610119565b91505092915050565b61015981610102565b82525050565b600060408201905061017460008301856100e4565b6101816020830184610150565b9392505050565b600081519050919050565b600081905092915050565b60005b838110156101c25780820151818401526020810190506101a7565b838111156101d1576000848401525b50505050565b60006101e2826101a4565b6101ec81856101af565b93506101fc8185602086016101a4565b80840191505092915050565b6000610214826101d7565b9150819050919050565b61022781610208565b82525050565b6000602082019050610242600083018461021e565b9291505056fea26469706673582212209b5c7f4b1b5c7f4b1b5c7f4b1b5c7f4b1b5c7f4b1b5c7f4b1b5c7f4b1b5c7f4b64736f6c63430008110033"
        }
        
        # Mock block hashes for the last 256 blocks
        self.block_hashes = {}
        for i in range(max(0, block_number - 256), block_number + 1):
            # Generate block hashes
            hash_int = (0x1234567890abcdef * i) % (2**256)
            self.block_hashes[i] = f"0x{hash_int:064x}"
    
    def get_account(self, address: str, expected_nonce: Optional[int] = None) -> Optional[AccountInfo]:
        """Get mock account information
        
        Args:
            address: The account address
            expected_nonce: If provided, the account will have this nonce (for simulation compatibility)
        """
        
        # If expected_nonce is provided, use it, otherwise use a deterministic nonce
        if expected_nonce is not None:
            nonce = expected_nonce
        else:
            addr_hash = hash(address) % 1000  # Generate nonce between 0-999
            nonce = max(0, addr_hash)
        
        # return a default account with some ETH
        return AccountInfo(
            balance=10 * 10**18,
            nonce=nonce,
            bytecode_hash="0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"  # empty account
        )
    
    def get_storage(self, address: str, slot: str) -> str:
        """Get mock storage value"""
        address = address.lower()
        key = f"{address}:{slot}"
        return self.storage.get(key, "0x0000000000000000000000000000000000000000000000000000000000000000")
    
    def get_code(self, address: str) -> str:
        """Get mock contract code"""
        address = address.lower()
        return self.contract_code.get(address, "0x")
    
    def get_block_hash(self, block_number: int) -> Optional[str]:
        """Get mock block hash"""
        return self.block_hashes.get(block_number)


class MockStateProviderFactory(StateProviderFactory):
    """Factory for creating mock state providers"""
    
    def __init__(self, default_block_number: int = 20114954):
        self.default_block_number = default_block_number
        self.chain_id = 1  # Ethereum mainnet
    
    def latest(self) -> StateProvider:
        """Get state provider for the latest (mock) block"""
        return MockStateProvider(self.default_block_number)
    
    def history_by_block_number(self, block_number: int) -> StateProvider:
        """Get state provider for a specific block number"""
        return MockStateProvider(block_number)
    
    def history_by_block_hash(self, block_hash: str) -> StateProvider:
        """Get state provider for a specific block hash"""
        # For mock, we ignore the hash and use default block
        return MockStateProvider(self.default_block_number)
    
    def get_simulation_context(self, block_number: int) -> SimulationContext:
        """Get simulation context for a block"""
        provider = MockStateProvider(block_number)
        return SimulationContext(
            block_number=block_number,
            block_timestamp=1672531200 + (block_number - 20114954) * 12,
            block_base_fee=5_000_000_000,
            block_gas_limit=36_000_000,
            block_hash=provider.block_hashes.get(block_number, "0x0"),
            parent_hash=provider.block_hashes.get(block_number - 1, "0x0"),
            chain_id=self.chain_id,
            coinbase="0x95222290DD7278Aa3Ddd389Cc1E1d165CC4BAfe5"  # Mock builder coinbase
        )
