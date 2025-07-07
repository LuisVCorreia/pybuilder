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
    
    # Additional fields for proper block header construction
    block_difficulty: int = 0  # PoS era, always 0
    block_gas_used: int = 0  # Starting gas used
    withdrawals_root: Optional[str] = None  # For post-Shanghai blocks
    blob_gas_used: Optional[int] = None  # For post-Cancun blocks 
    excess_blob_gas: Optional[int] = None  # For post-Cancun blocks
    
    @classmethod
    def from_onchain_block(cls, onchain_block: dict, winning_bid_trace: dict = None) -> 'SimulationContext':
        """
        Create SimulationContext from onchain block data.
        This mirrors rbuilder's BlockBuildingContext::from_onchain_block()
        """
        
        # Extract all the fields we need for proper block header construction
        context = cls(
            block_number=onchain_block.get('number', 0),
            block_timestamp=onchain_block.get('timestamp', 0),
            block_base_fee=onchain_block.get('baseFeePerGas', 0),
            block_gas_limit=onchain_block.get('gasLimit', 36000000),
            block_hash=onchain_block.get('hash'),
            parent_hash=onchain_block.get('parentHash'),
            chain_id=1,  # Ethereum mainnet
            coinbase=onchain_block.get('miner', "0x0000000000000000000000000000000000000000"),
            block_difficulty=onchain_block.get('difficulty', 0),
            block_gas_used=onchain_block.get('gasUsed', 0),
            withdrawals_root=onchain_block.get('withdrawalsRoot'),
            blob_gas_used=onchain_block.get('blobGasUsed'),
            excess_blob_gas=onchain_block.get('excessBlobGas')
        )

        # If we have winning bid trace, use the fee recipient from there
        # This matches rbuilder's logic of using suggested_fee_recipient
        if winning_bid_trace and 'proposer_fee_recipient' in winning_bid_trace:
            context.coinbase = winning_bid_trace['proposer_fee_recipient']
        
        return context


class StateProvider(ABC):
    """Abstract interface for accessing Ethereum state at a specific block"""
    
    @abstractmethod
    def get_account(self, address: str) -> Optional[AccountInfo]:
        """Get account information (balance, nonce, code hash, storage root)
        
        Args:
            address: The account address  
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
    def create_simulation_context(self, block_number: int) -> SimulationContext:
        """Create simulation context for a block"""
        pass
