import os
import yaml

def load_config(path: str) -> dict:
    with open(os.path.expanduser(path)) as f:
        cfg = yaml.safe_load(f)
    # expand both ~ and ${VARS}
    cfg["mempool_data_dir"] = os.path.expanduser(cfg["mempool_data_dir"])
    cfg["sqlite_db_path"]   = os.path.expanduser(cfg["sqlite_db_path"])
    cfg["provider_url"]     = os.path.expandvars(cfg["provider_url"])
    cfg["logging_level"]    = cfg.get("logging_level", "INFO")
    return cfg
