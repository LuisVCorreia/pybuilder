import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class NonceCache:
    """
    Simple nonce cache for tracking account nonces during block building.
    """
    
    def __init__(self, state_provider: Any):
        """
        Initialize nonce cache with state provider.
        
        Args:
            state_provider: State provider for fetching on-chain nonces
        """
        self.state_provider = state_provider
        self.cached_nonces: Dict[str, int] = {}
    
    def nonce(self, address: str) -> int:
        """
        Get the current nonce for an address.
        
        Args:
            address: Account address (hex string)
            
        Returns:
            Current nonce for the account
        """
        if address not in self.cached_nonces:
            # In a real implementation, this would fetch from state_provider
            # For now, we'll default to 0 for simplicity
            self.cached_nonces[address] = self._fetch_onchain_nonce(address)
        
        return self.cached_nonces[address]
    
    def _fetch_onchain_nonce(self, address: str) -> int:
        """
        Fetch on-chain nonce for an address.
        
        This is a simplified implementation. In rbuilder, this would
        use the StateProvider to get the actual on-chain nonce.
        """
        # TODO: Implement actual nonce fetching from state
        # For backtesting, we can extract nonces from the simulation context
        return 0
    
    def update_nonce(self, address: str, nonce: int) -> None:
        """
        Update cached nonce for an address.
        
        Args:
            address: Account address
            nonce: New nonce value
        """
        self.cached_nonces[address] = nonce
    
    def get_cached_nonces(self) -> Dict[str, int]:
        """Get all cached nonces."""
        return self.cached_nonces.copy()
