from __future__ import annotations
import pickle
from pathlib import Path
from typing import List, Optional, Type
import threading
import logging

from boa.rpc import RPC, to_int, to_hex
from boa.vm.fork import AccountDBFork, CachingRPC, _PREDEFINED_BLOCKS
from boa.vm.py_evm import PyEVM, VMPatcher, titanoboa_computation, Sha3PreimageTracer, SstoreTracer, GENESIS_PARAMS
from boa.util.sqlitedb import SqliteCache
from eth.db.account import AccountDB
from boa.vm.fast_accountdb import patch_pyevm_state_object
import eth.tools.builder.chain as chain
from eth.chains.mainnet import MainnetChain
from eth.db.atomic import AtomicDB


CHAIN_ID = 1  # Hardcoded chain id (Ethereum mainnet)
PREV_HASH_WINDOW = 256
ENABLE_PREV_HASHES = True
THREADSAFE_CACHE = True
BLOCK_HASH_KEY_PREFIX = "boa_ext_prev_hashes"  # Prefix for aggregated prev-hash cache entries

logger = logging.getLogger("boa_ext")

_ALREADY_PATCHED = False
_PATCH_LOCK = threading.Lock()

def apply_patches():
    global _ALREADY_PATCHED
    with _PATCH_LOCK:
        if _ALREADY_PATCHED:
            return
        _patch_sqlite_cache()
        _patch_caching_rpc_new()
        _patch_fetch_uncached()
        _patch_account_db_fork()
        _patch_pyevm_fork_rpc()
        _patch_init_vm()
        _patch_make_chain()
        _ALREADY_PATCHED = True
        logger.debug("boa_ext patches applied.")

def _patch_sqlite_cache():
    if not THREADSAFE_CACHE:
        return

    # Keep one SqliteCache per (path, thread), not just per path.
    instances = {}
    orig_create = SqliteCache.create

    def create_per_path_thread(cls, db_path, *a, **k):
        p = str(Path(db_path).resolve())
        tid = threading.get_ident()
        key = (p, tid)
        inst = instances.get(key)
        if inst is None:
            inst = cls.__new__(cls)
            cls.__init__(inst, p, *a, **k)  # new sqlite3.Connection in this thread
            instances[key] = inst
        return inst

    SqliteCache.create = classmethod(create_per_path_thread)
    SqliteCache._boa_ext_original_create = orig_create

def _patch_caching_rpc_new():
    orig_new = CachingRPC.__new__
    def new(cls, rpc, chain_id, debug):
        import threading
        thread_id = threading.get_ident()
        # replicate original but add thread id
        if isinstance(rpc, cls) and rpc._chain_id == chain_id:
            return rpc

        if (rpc.identifier, chain_id, thread_id) in cls._loaded:
            return cls._loaded[(rpc.identifier, chain_id, thread_id)]

        ret = super(CachingRPC, cls).__new__(cls)
        ret.__init__(rpc, chain_id, debug)
        cls._loaded[(rpc.identifier, chain_id, thread_id)] = ret
        return ret
    CachingRPC.__new__ = new



def _patch_fetch_uncached():
    def _redirect(self: RPC, method: str, params):
        # Always use cached path
        return self.fetch(method, params)
    RPC.fetch_uncached = _redirect

def _patch_account_db_fork():
    """
    Patch AccountDBFork so:
      - class_from_rpc does not use fetch_uncached (hardcodes chain_id = CHAIN_ID).
      - __init__ uses only a single cached fetch for the block header.
    """
    def patched_class_from_rpc(cls, rpc: RPC, block_identifier, debug: bool, **kwargs):
        chain_id = CHAIN_ID  # hardcoded mainnet

        class _ConfiguredAccountDB(AccountDBFork):
            def __init__(self, *a, **k):
                caching_rpc = CachingRPC(rpc, chain_id, debug, **kwargs)
                super().__init__(caching_rpc, chain_id, block_identifier, *a, **k)

        return _ConfiguredAccountDB

    AccountDBFork.class_from_rpc = classmethod(patched_class_from_rpc)

    def patched_init(self, rpc: CachingRPC, chain_id: int, block_identifier, *a, **k):
        # Call only AccountDB.__init__ (super chain). We cannot easily
        # import AccountDB symbolically here, but super() works because
        # MRO: AccountDBFork -> AccountDB -> ...
        super(AccountDBFork, self).__init__(*a, **k)

        from eth.db.backends.memory import MemoryDB  # safe import here
        from eth.db.journal import JournalDB

        self._dontfetch = JournalDB(MemoryDB())
        self._rpc = rpc

        # Normalize block identifier
        if block_identifier not in _PREDEFINED_BLOCKS:
            if isinstance(block_identifier, int):
                block_identifier_hex = hex(block_identifier)
            elif isinstance(block_identifier, str):
                block_identifier_hex = (
                    block_identifier if block_identifier.startswith("0x") else to_hex(block_identifier)
                )
            else:
                block_identifier_hex = to_hex(block_identifier)
        else:
            block_identifier_hex = block_identifier

        self._chain_id = chain_id

        # Single cached fetch (no fetch_uncached)
        self._block_info = self._rpc.fetch("eth_getBlockByNumber", [block_identifier_hex, False])
        self._block_number = to_int(self._block_info["number"])

    AccountDBFork.__init__ = patched_init

def _patch_init_vm():

    def patched_init_vm(self, account_db_class=AccountDB, block_number: Optional[int] = None):
        head = self.chain.get_canonical_head()
        if block_number is None:
            target_header = head
        else:
            # cheap header: copy() only mutates fields we pass
            target_header = head.copy(block_number=block_number)

        self.vm = self.chain.get_vm(target_header)
        self.vm.__class__._state_class.account_db_class = account_db_class

        # patcher & computation class wiring (unchanged)
        self.patch = VMPatcher(self.vm)

        c: Type[titanoboa_computation] = type(
            "TitanoboaComputation",
            (titanoboa_computation, self.vm.state.computation_class),
            {"env": self.env},
        )

        if self._fast_mode_enabled:
            patch_pyevm_state_object(self.vm.state)

        self.vm.state.computation_class = c
        c.opcodes[0x20] = Sha3PreimageTracer(c.opcodes[0x20], self.env)
        c.opcodes[0x55] = SstoreTracer(c.opcodes[0x55], self.env)

    PyEVM._init_vm = patched_init_vm

def _patch_pyevm_fork_rpc():

    def patched_fork_rpc(self: PyEVM, rpc: RPC, block_identifier: str, debug: bool, **kwargs):
        # Perform original fork (builds AccountDBFork etc.)
        account_db_class = AccountDBFork.class_from_rpc(
            rpc, block_identifier, debug, **kwargs
        )

        current_block = int(block_identifier, 16)
        self._init_vm(account_db_class=account_db_class, block_number=current_block)

        block_info = self.vm.state._account_db._block_info
        chain_id = self.vm.state._account_db._chain_id

        # 4) patch execution-context values
        self.patch.timestamp    = int(block_info["timestamp"], 16)
        self.patch.block_number = int(block_info["number"], 16)
        self.patch.chain_id     = chain_id

        if not ENABLE_PREV_HASHES:
            return

        # Ensure we actually have a forked AccountDBFork underneath
        try:
            account_db = self.vm.state._account_db
        except Exception:
            return

        # Guard: require the patched account_db type
        try:
            if not self.is_forked:
                return
        except Exception:
            pass

        caching_rpc: CachingRPC = account_db._rpc
        fork_block_number = int(account_db._block_info["number"], 16)

        prev_hashes = _load_or_build_prev_hashes(self, caching_rpc, fork_block_number)
        # Assign to execution context (parent first)
        self.patch.prev_hashes = prev_hashes

    PyEVM.fork_rpc = patched_fork_rpc

def _patch_make_chain():
    import boa.vm.py_evm as py_mod  # patch module, not class

    def patched_make_chain():
        # start with all mainnet forks
        print("Patching make_chain to use MainnetChain with GENESIS_PARAMS")
        _Full = chain.build(MainnetChain)

        full_cfg = _Full.vm_configuration
        ChainCls = _Full.configure(vm_configuration=full_cfg)
        print("Full cfg:", full_cfg)
        return ChainCls.from_genesis(AtomicDB(), GENESIS_PARAMS)

    py_mod._make_chain = patched_make_chain


def _cache_key(block_number: int) -> bytes:
    # Key layout includes chain id and block number
    return pickle.dumps((BLOCK_HASH_KEY_PREFIX, CHAIN_ID, block_number))


def _load_or_build_prev_hashes(evm: PyEVM, caching_rpc: CachingRPC, fork_block_number: int) -> List[bytes]:
    """
    Returns list of previous block hashes for BLOCKHASH opcode:
      index 0 = hash(fork_block_number - 1)
      index 1 = hash(fork_block_number - 2)
      ...
      up to PREV_HASH_WINDOW or genesis boundary.
    Persisted as a single value in titanoboa's existing sqlite-backed cache.
    """
    db = getattr(caching_rpc, "_db", None)
    key = _cache_key(fork_block_number)

    if db is not None:
        try:
            return pickle.loads(db[key])
        except KeyError:
            pass  # need to build

    hashes: List[bytes] = []  # parent hash first
    # Starting parent block number (parent hash in block block_number is in block fork_block_number)
    start = fork_block_number
    end = max(-1, fork_block_number - PREV_HASH_WINDOW)  # inclusive stop logic with range()

    for bn in range(start, end, -1):
        if bn < 0:
            break
        try:
            block_info = caching_rpc.fetch("eth_getBlockByNumber", [hex(bn), False])
            h = bytes.fromhex(block_info["hash"].removeprefix("0x"))
        except Exception:
            # If a fetch fails, append zero hash (match your earlier fallback)
            h = b"\x00" * 32
        hashes.append(h)

    # Cache the *variable-length* list (length = min(PREV_HASH_WINDOW, fork_block_number))
    if db is not None:
        db[key] = pickle.dumps(hashes)

    return hashes
