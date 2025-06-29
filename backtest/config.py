import os
import yaml

def load_config(path: str) -> dict:
    with open(os.path.expanduser(path)) as f:
        cfg = yaml.safe_load(f)
    # expand both ~ and ${VARS}
    cfg["fetch_mempool_data_dir"] = os.path.expanduser(cfg["fetch_mempool_data_dir"])
    cfg["fetch_sqlite_db_path"]   = os.path.expanduser(cfg["fetch_sqlite_db_path"])
    cfg["fetch_rpc_url"]     = os.path.expandvars(cfg["fetch_rpc_url"])
    cfg["logging_level"]    = cfg.get("logging_level", "INFO")
    cfg["fetch_concurrency_limit"] = cfg.get("fetch_concurrency_limit", 1)
    return cfg
