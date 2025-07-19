"""
Importing this package applies Titanoboa / py-evm monkey patches:
- Thread-safe per-path SqliteCache
- Hardcoded chain id (1) removal of remote eth_chainId calls
- Cached previous block hashes (persisted)
- Removal of uncached RPC fetch usage
"""

from .patches import apply_patches  # noqa: F401

# Immediately apply patches on import
apply_patches()