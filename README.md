# pybuilder

This project is a Python adaptation of Flashbots' [rbuilder](https://github.com/flashbots/rbuilder). Unlike rbuilder, which is designed to build blocks in real time, this repository focuses solely on backtesting. It does not implement live building capabilities.

## Usage

The workflow consists of two steps:

1. **Fetch a block and its mempool transactions.**
   Run the fetch script specifying the block number:
   ```bash
   python scripts/run_backtest_fetch.py --block 20114954
   ```
   This downloads the relevant mempool data using Flashbots' [Mempool Dumpster](https://mempool-dumpster.flashbots.net/index.html) and stores everything in a SQLite database configured in `config.yaml`.

2. **Build the block.**
   Building logic is not implemented yet. Future versions will replay orders from the database to construct blocks for analysis.