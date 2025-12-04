import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import logging

def main():

    from backtest.config import load_config
    from backtest.build.build_block import run_backtest
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Run a backtest to build a block from historical data.")
    
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--only-order-ids", nargs='+', help="Use only these order IDs")
    parser.add_argument("--block-building-time-ms", type=int, default=0, help="Block building lag in milliseconds")
    parser.add_argument("--show-sim", action="store_true", help="Show order data and top of block simulation results")
    parser.add_argument("--no-block-building", action="store_true", help="Skip the block building process")
    parser.add_argument("--no-conflict-resolution", action="store_true", help="Disable conflict resolution in parallel builder")
    parser.add_argument("--builders", nargs='+', default=[], help="Builders to build block with")
    parser.add_argument("block", type=int, help="Block number to build")


    args = parser.parse_args()
    
    config = load_config(args.config)

    # Set up logging from config
    log_level = getattr(logging, config.get("logging_level", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    run_backtest(args, config)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.shutdown()
        sys.exit(0)
