import os
import yaml
import logging

logger = logging.getLogger(__name__)

def load_config(path: str) -> dict:
    with open(os.path.expanduser(path)) as f:
        cfg = yaml.safe_load(f)

    cfg["fetch_mempool_data_dir"] = os.path.expanduser(cfg["fetch_mempool_data_dir"])
    cfg["fetch_sqlite_db_path"] = os.path.expanduser(cfg["fetch_sqlite_db_path"])
    cfg["build_cache_path"] = os.path.expanduser(cfg["build_cache_path"])
    cfg["fetch_rpc_url"] = os.path.expandvars(cfg["fetch_rpc_url"])
    cfg["logging_level"] = cfg.get("logging_level", "INFO")
    cfg["batch_size"] = cfg.get("fetch_batch_size", 1)

    _validate_builders(cfg)
    
    return cfg


def _validate_builders(config: dict) -> None:
    builders = config.get('builders', [])
    
    if not builders:
        logger.warning("No builders configured")
        return
    
    valid_algos = {'ordering-builder', 'parallel-builder'}
    valid_sortings = {'max-profit', 'mev-gas-price'}
    
    for builder in builders:
        name = builder.get('name', 'unnamed')
        algo = builder.get('algo')
        
        # Validate algorithm
        if algo not in valid_algos:
            raise ValueError(f"Builder {name}: invalid algo '{algo}'. Must be one of: {valid_algos}")
        
        # Validate ordering builder specific fields
        if algo == 'ordering-builder':
            sorting = builder.get('sorting')
            if sorting not in valid_sortings:
                raise ValueError(f"Builder {name}: invalid sorting '{sorting}'. Must be one of: {valid_sortings}")
        
        logger.debug(f"Validated builder: {name} ({algo})")
