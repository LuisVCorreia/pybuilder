# pybuilder

pybuilder is a Python-based backtesting framework for Ethereum block building, adapted from Flashbots' [rbuilder](https://github.com/flashbots/rbuilder). It removes the requirement for a local archive node and enables users to simulate the block building process using a standard RPC endpoint (e.g., [Alchemy](https://www.alchemy.com/)).


## Setup

### 1. Prerequisites
You will need access to an Ethereum node provider to fetch chain data.
1. Create an account or sign in to [Alchemy](https://www.alchemy.com/).
2. Create a new App (Ethereum Mainnet) and copy the **API Key**.
3. Create a `.env` file in the root of the project and add your key:

```bash
ALCHEMY_API_KEY="{YOUR_ALCHEMY_API_KEY}"
```

### 2. Installation
To install the project dependencies, run the following:
```bash
# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

The workflow consists of two steps:

### 1. **Fetch Historical Data.**
Fetch the state and all pending mempool transactions that were available before the block we're backtesting was proposed. The mempool data is downloaded using Flashbots' [Mempool Dumpster](https://mempool-dumpster.flashbots.net/index.html) and stored in an SQLite database.

```bash
python scripts/run_backtest_fetch.py --block 20757091
```

### 2. **Run Block Building Simulation.**
Use the fetched data to simulate block construction. This enables you to test different ordering algorithms and benchmark your results against the actual block that landed on-chain.
```bash
python scripts/run_backtest_build.py 20757091
```
