# src/main.py

import time
import yaml

from client.futures_client import FuturesClient
from utils.position_manager import PositionManager
from utils.risk_management import RiskManager

from strategies.mean_reversion import MeanReversionStrategy
from strategies.grid_strategy import GridStrategy
# from strategies.ml_strategy import MLStrategy  # uncomment when ML is ready

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # 1) Load configuration (includes exchange.api_key, api_secret, testnet flag)
    cfg = load_config()

    # 2) Instantiate Binance Futures client
    client = FuturesClient(cfg)

    # 3) Portfolio‐Level Risk Manager (correlation & allocation)
    rm = RiskManager(cfg, data_dir="data/klines")
    symbols = [s for s, v in cfg["symbols"].items() if v.get("enabled", False)]
    base_allocs = {s: cfg["symbols"][s].get("allocation_pct", 0) for s in symbols}

    # 3a) Adjust for correlations & net exposure
    adjusted = rm.adjust_allocations(symbols, base_allocs, corr_threshold=0.8, reduction_pct=0.5)
    final_allocs = rm.enforce_notional_cap(adjusted, max_net_pct=50)
    for s in symbols:
        cfg["symbols"][s]["allocation_pct"] = final_allocs.get(s, 0)

    # 4) PositionManager: one shared instance
    pm = PositionManager("state/positions.json")
    pm.reconcile(client)

    # 5) Build strategy instances
    strategies = []
    for symbol in symbols:
        sym_cfg = cfg["symbols"][symbol]
        strat_name = sym_cfg["strategy"].lower()

        if strat_name == "grid":
            strat = GridStrategy(client, cfg, symbol, pm)
        elif strat_name == "mean_reversion":
            strat = MeanReversionStrategy(client, cfg, symbol, pm)
        # elif strat_name == "ml":
        #     strat = MLStrategy(client, cfg, symbol, pm)

        else:
            continue

        strategies.append(strat)

    # 6) Main loop: invoke each strategy
    while True:
        for strat in strategies:
            try:
                strat.run()
            except Exception as e:
                print(f"[ERROR] {strat.symbol} ({strat.__class__.__name__}): {e}")
        time.sleep(60)

if __name__ == "__main__":
    main()
