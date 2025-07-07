"""Current implementation uses Titanaboa's rpc.py to interact with an Alchemy node, not this."""
import logging
from typing import Optional, Dict, Any
from web3 import Web3
from eth_utils.address import to_checksum_address
from eth_utils.conversions import to_hex

from .state_provider import StateProvider, StateProviderFactory, AccountInfo
from .evm_simulator import SimulationContext

logger = logging.getLogger(__name__)


class AlchemyStateProvider(StateProvider):
    """State provider that fetches data from Alchemy RPC endpoint"""
    
    def __init__(self, rpc_url: str, block_number: int):
        self.rpc_url = rpc_url
        self.block_number = block_number
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self._cache = {}  # Simple cache for frequently accessed data
        
        # Validate connection
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to Alchemy RPC at {rpc_url}")
        
        logger.info(f"Connected to Alchemy RPC at block {block_number}")

    def get_account(self, address: str) -> Optional[AccountInfo]:
        """Get account information from Alchemy RPC
        
        Args:
            address: The account address
        """
        try:
            address = to_checksum_address(address)
            cache_key = f"account_{address}_{self.block_number}"
            
            if cache_key in self._cache:
                cached_account = self._cache[cache_key]
                
                return cached_account
            
            # Get balance and nonce from RPC
            balance = self.w3.eth.get_balance(address, block_identifier=self.block_number)
            nonce = self.w3.eth.get_transaction_count(address, block_identifier=self.block_number)
            
            logger.info(f"DEBUGGING: Alchemy RPC returned balance={balance} wei for {address} at block {self.block_number}")
            
            # Get code to determine bytecode hash
            code = self.w3.eth.get_code(address, block_identifier=self.block_number)
            print("code returned:", code.hex())
            bytecode_hash = Web3.keccak(code).hex() if code else "0x" + "0" * 64
            
            account_info = AccountInfo(
                balance=balance,
                nonce=nonce,
                bytecode_hash=bytecode_hash
            )

            self._cache[cache_key] = account_info
            
            return account_info
            
        except Exception as e:
            logger.error(f"Failed to get account {address}: {e}")
            return None
    
    def get_storage(self, address: str, slot: str) -> str:
        """Get storage value at specific slot"""
        try:
            address = to_checksum_address(address)
            cache_key = f"storage_{address}_{slot}_{self.block_number}"
            
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            # Convert slot to proper format
            if not slot.startswith('0x'):
                slot = '0x' + slot
            
            storage_value = self.w3.eth.get_storage_at(
                address, 
                int(slot, 16), 
                block_identifier=self.block_number
            )
            
            result = to_hex(storage_value)
            self._cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Failed to get storage {address}[{slot}]: {e}")
            return "0x" + "0" * 64
    
    def get_code(self, address: str) -> str:
        """Get contract code"""
        try:
            address = to_checksum_address(address)
            cache_key = f"code_{address}_{self.block_number}"
            
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            code = self.w3.eth.get_code(address, block_identifier=self.block_number)
            result = to_hex(code)
            
            self._cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Failed to get code for {address}: {e}")
            return "0x"
    
    def get_block_hash(self, block_number: int) -> Optional[str]:
        """Get block hash for given block number"""
        try:
            cache_key = f"block_hash_{block_number}"
            
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            block = self.w3.eth.get_block(block_number)
            result = block['hash'].hex() if block else None
            
            if result:
                self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get block hash for {block_number}: {e}")
            return None
    
    def get_block_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed block information"""
        try:
            cache_key = f"block_info_{self.block_number}"
            
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            block = self.w3.eth.get_block(self.block_number)
            if not block:
                return None
            
            result = {
                'number': block['number'],
                'hash': block['hash'].hex(),
                'parent_hash': block['parentHash'].hex(),
                'timestamp': block['timestamp'],
                'gas_limit': block['gasLimit'],
                'gas_used': block['gasUsed'],
                'base_fee': block.get('baseFeePerGas', 0),
                'difficulty': block['difficulty'],
                'coinbase': block['miner']
            }
            
            self._cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Failed to get block info for {self.block_number}: {e}")
            return None


class AlchemyStateProviderFactory(StateProviderFactory):
    """Factory for creating Alchemy state providers"""
    
    def __init__(self, rpc_url: str):
        self.rpc_url = rpc_url
        
        # Validate connection on creation
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.is_connected():
            raise ConnectionError(f"Failed to connect to Alchemy RPC at {rpc_url}")
        
        logger.info(f"Alchemy state provider factory initialized with URL: {rpc_url}")
    
    def latest(self) -> StateProvider:
        """Get state provider for the latest block"""
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        latest_block = w3.eth.get_block('latest')
        return AlchemyStateProvider(self.rpc_url, latest_block['number'])
    
    def history_by_block_number(self, block_number: int) -> StateProvider:
        """Get state provider for a specific block number"""
        return AlchemyStateProvider(self.rpc_url, block_number)
    
    def history_by_block_hash(self, block_hash: str) -> StateProvider:
        """Get state provider for a specific block hash"""
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        block = w3.eth.get_block(block_hash)
        return AlchemyStateProvider(self.rpc_url, block['number'])
    
    def create_simulation_context(self, block_number: int) -> SimulationContext:
        """Create simulation context from Alchemy block data"""
        # Get block info directly from the provider
        block_info = {
            'number': block_number,
            'hash': '0x' + '0' * 64,  # Placeholder
            'parent_hash': '0x' + '0' * 64,  # Placeholder
            'timestamp': 0,  # Placeholder
            'base_fee': 0,  # Placeholder
            'gas_limit': 30000000,  # Default
            'coinbase': '0x' + '0' * 40  # Placeholder
        }
        
        if not block_info:
            raise ValueError(f"Failed to get block info for block {block_number}")
        
        return SimulationContext(
            block_number=block_info['number'],
            block_timestamp=block_info['timestamp'],
            block_base_fee=block_info['base_fee'],
            block_gas_limit=block_info['gas_limit'],
            block_hash=block_info['hash'],
            parent_hash=block_info['parent_hash'],
            chain_id=1,  # Ethereum mainnet
            coinbase=block_info['coinbase']
        )
