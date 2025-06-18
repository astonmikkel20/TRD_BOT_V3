# run_backtest.py (in project root)

import os
import yaml
import pandas as pd

from src.backtesting.backtester import Backtester

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main():
    cfg = load_config()
    hist_dir = os.path.join("data", "klines")
    out_dir = os.path.join("backtesting", "default_results")
    ensure_dir(out_dir)

    any_run = False
    results = []

    for symbol, sym_cfg in cfg["symbols"].items():
        if not sym_cfg.get("enabled", False):
            continue

        strat_name = sym_cfg.get("strategy", "").lower()
        if strat_name not in ["grid", "mean_reversion", "ml"]:
            continue

        any_run = True
        print(f"Backtesting {symbol} with {strat_name}…")
        bt = Backtester(symbol=symbol, config=cfg, data_dir=hist_dir, strategy_name=strat_name)
        metrics = bt.run()
        print(f"→ {symbol} metrics: {metrics}\n")

        # Save per-symbol files
        df_trades = pd.DataFrame(bt.trades)
        df_trades.to_csv(os.path.join(out_dir, f"trades_{symbol}.csv"), index=False)

        equity_df = pd.DataFrame({
            "timestamp": pd.to_datetime(bt.df["open_time"]),
            "equity": bt.equity_curve
        })
        equity_df.to_csv(os.path.join(out_dir, f"equity_{symbol}.csv"), index=False)

        results.append({
            "symbol": symbol,
            "strategy": strat_name,
            **metrics
        })

    if not any_run:
        print("⚠️  No enabled symbols found in config.yaml – nothing to backtest.")
        return

    # Write a summary
    df_summary = pd.DataFrame(results)
    df_summary.to_csv(os.path.join(out_dir, "summary_all_symbols.csv"), index=False)
    print(f"\n✅ Backtest complete. Results saved in {out_dir}/")

if __name__ == "__main__":
    main()
