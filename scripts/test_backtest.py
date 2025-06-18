# scripts/test_backtest.py

import os, sys

# 1) Ensure project root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.backtesting.backtester import Backtester

# 2) Data directory where your downloaded klines live
DATA_DIR = "data/klines"

# 3) Minimal config for XRPUSDT
cfg = {
  "capital_usdt": 1000,
  "symbols": {
    "XRPUSDT": {
      "contract_type": "PERPETUAL",
      "strategy": "grid",
      "grid": {
        "vol_lookback": 1000,
        "vol_multiplier": 1.0,
        "base_spacing_pct": 0.01
      }
    }
  },
  "defaults": {"leverage": 1, "position_mode": "ONE_WAY"},
  "risk_defaults": {
    "max_position_size_pct": 100,
    "stop_loss_pct": 0.05,
    "take_profit_pct": 0.15
  }
}

# 4) Run backtest on XRPUSDT
bt = Backtester(symbol="XRPUSDT", config=cfg, data_dir=DATA_DIR, strategy_name="grid")
metrics = bt.run()
print("Backtest metrics for XRPUSDT (grid):", metrics)
