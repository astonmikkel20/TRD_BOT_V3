import yaml
import itertools
import pandas as pd
from backtesting.backtester import Backtester

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def generate_param_grid():
    """
    Generate all combinations of:
      lookback: [20, 50, 80]
      std_dev_multiplier: [1.5, 2.0, 2.5]
      rsi_period: [14, 21]
      rsi_oversold: [30, 25]
      rsi_overbought: [70, 80]
      trend_lookback: [50, 100]
    Returns a list of dicts.
    """
    lookbacks = [20, 50, 80]
    std_mults = [1.5, 2.0, 2.5]
    rsi_periods = [14, 21]
    rsi_ov = [30, 25]
    rsi_ob = [70, 80]
    trend_lb = [50, 100]

    grid = []
    for lb, sm, rp, rov, rob, tlb in itertools.product(
        lookbacks, std_mults, rsi_periods, rsi_ov, rsi_ob, trend_lb
    ):
        grid.append({
            "lookback": lb,
            "std_dev_multiplier": sm,
            "rsi_period": rp,
            "rsi_oversold": rov,
            "rsi_overbought": rob,
            "trend_lookback": tlb
        })
    return grid

def main():
    # 1) Load base config
    cfg = load_config("config/config.yaml")

    # 2) Identify which symbols use mean_reversion
    symbols = []
    for sym, sym_cfg in cfg["symbols"].items():
        if sym_cfg.get("enabled", False) and sym_cfg.get("strategy", "") == "mean_reversion":
            symbols.append(sym)

    # 3) Path to historical OHLC
    hist_data_dir = "data/klines"  # e.g. contains ETHUSDT_1h.csv, BTCUSDT_1h.csv, etc.

    param_grid = generate_param_grid()
    all_results = []

    for symbol in symbols:
        print(f"\n=== Scanning hyperparameters for {symbol} ===")
        results = []

        for params in param_grid:
            # Build a modified config for this test
            test_cfg = dict(cfg)  # shallow copy of top‐level
            sym_block = test_cfg["symbols"][symbol]

            # Inject override keys
            sym_block["lookback_override"] = params["lookback"]
            sym_block["std_dev_multiplier_override"] = params["std_dev_multiplier"]
            sym_block["rsi_period_override"] = params["rsi_period"]
            sym_block["rsi_oversold_override"] = params["rsi_oversold"]
            sym_block["rsi_overbought_override"] = params["rsi_overbought"]
            sym_block["trend_lookback_override"] = params["trend_lookback"]

            # Run backtester
            bt = Backtester(
                symbol=symbol,
                config=test_cfg,
                data_dir=hist_data_dir,
                strategy_name="mean_reversion"
            )
            metrics = bt.run()
            # Record both the params and the resulting performance
            row = dict(symbol=symbol, **params, **metrics)
            results.append(row)

        # Save per‐symbol results to CSV
        df_res = pd.DataFrame(results)
        out_path = f"backtesting/results_{symbol}.csv"
        df_res.to_csv(out_path, index=False)
        print(f"Saved results to {out_path}")

        all_results.extend(results)

    # Optionally save everything combined
    df_all = pd.DataFrame(all_results)
    df_all.to_csv("backtesting/all_results.csv", index=False)
    print("Hyperparameter scan complete. Combined results in backtesting/all_results.csv")


if __name__ == "__main__":
    main()
