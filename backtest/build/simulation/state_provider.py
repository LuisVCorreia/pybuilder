from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass


@dataclass
class AccountInfo:
    """Ethereum account information"""
    balance: int  # in wei
    nonce: int
    bytecode_hash: str  # 0x-prefixed hex


@dataclass
class SimulationContext:
    block_number: int
    block_timestamp: int  # Unix timestamp
    block_base_fee: int  # Base fee in wei
    block_gas_limit: int  # Block gas limit
    block_hash: Optional[str] = None
    parent_hash: Optional[str] = None
    chain_id: int = 1
    coinbase: str = "0x0000000000000000000000000000000000000000"  # fee recipient address


class StateProvider(ABC):
    """Abstract interface for accessing Ethereum state at a specific block"""
    
    @abstractmethod
    def get_account(self, address: str, expected_nonce: Optional[int] = None) -> Optional[AccountInfo]:
        """Get account information (balance, nonce, code hash, storage root)
        
        Args:
            address: The account address  
            expected_nonce: Optional hint for mock providers to return an account with this nonce
        """
        pass
    
    @abstractmethod
    def get_storage(self, address: str, slot: str) -> str:
        """Get storage value at specific slot. Returns 0x-prefixed hex string"""
        pass
    
    @abstractmethod
    def get_code(self, address: str) -> str:
        """Get contract code. Returns 0x-prefixed hex string"""
        pass
    
    @abstractmethod
    def get_block_hash(self, block_number: int) -> Optional[str]:
        """Get block hash for given block number"""
        pass


class StateProviderFactory(ABC):
    """Factory for creating state providers for specific blocks"""
    
    @abstractmethod
    def latest(self) -> StateProvider:
        """Get state provider for the latest block"""
        pass
    
    @abstractmethod
    def history_by_block_number(self, block_number: int) -> StateProvider:
        """Get state provider for a specific block number"""
        pass
    
    @abstractmethod
    def history_by_block_hash(self, block_hash: str) -> StateProvider:
        """Get state provider for a specific block hash"""
        pass
    
    @abstractmethod
    def get_simulation_context(self, block_number: int) -> SimulationContext:
        """Get simulation context for a block"""
        pass
