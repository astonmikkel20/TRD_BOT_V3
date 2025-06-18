# TRD_BOT_V3/src/backtesting/walkforward.py

import os
import yaml
import pandas as pd
from datetime import timedelta
from backtesting.backtester import Backtester
from backtesting.hyperscan import generate_param_grid

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def walk_forward(
    symbol: str,
    cfg: dict,
    data_dir: str = "data/klines",
    temp_dir: str = "data/temp",
    output_csv: str = "backtesting/walkforward_results.csv",
    train_months: int = 1,
    test_months: int = 1
):
    """
    Perform walk-forward on `symbol` using hourly data in data_dir/{symbol}_1h.csv.
    For each window:
      1) Train window = train_months months
      2) Test window = next test_months months
      3) On train: scan all hyperparameter combinations → pick best Sharpe
      4) On test: backtest that best parameter set → record metrics
    Saves a CSV of results to output_csv.
    """

    # 1) Load full hourly DataFrame
    hist_path = os.path.join(data_dir, f"{symbol}_1h.csv")
    if not os.path.isfile(hist_path):
        raise FileNotFoundError(f"Missing historical CSV: {hist_path}")
    df = pd.read_csv(hist_path, parse_dates=["open_time"])
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 2) Build train/test windows
    #    We'll advance by test_months each iteration.
    results = []
    start_time = df["open_time"].min()
    end_time = df["open_time"].max()

    cur_train_start = start_time

    # Helper to add months (approx by pandas DateOffset)
    from pandas.tseries.offsets import DateOffset

    while True:
        train_end = cur_train_start + DateOffset(months=train_months)
        test_start = train_end
        test_end = train_end + DateOffset(months=test_months)

        # Stop if test_end exceeds available data
        if test_end > end_time:
            break

        # Slice DataFrames
        df_train = df[(df["open_time"] >= cur_train_start) & (df["open_time"] < train_end)].copy()
        df_test = df[(df["open_time"] >= test_start) & (df["open_time"] < test_end)].copy()

        # Skip if insufficient data (e.g. less than 100 hourly bars)
        if len(df_train) < 100 or len(df_test) < 100:
            cur_train_start = test_start
            continue

        # 3) Save train/test to temp CSVs for the Backtester to use
        ensure_dir(temp_dir)
        train_file = os.path.join(temp_dir, f"{symbol}_train_{cur_train_start.date()}.csv")
        test_file = os.path.join(temp_dir, f"{symbol}_test_{test_start.date()}.csv")
        df_train.to_csv(train_file, index=False)
        df_test.to_csv(test_file, index=False)

        # 4) Hyperparameter scan on train set
        print(f"\n=== Walk-forward: Training {symbol} from {cur_train_start.date()} to {train_end.date()} ===")
        best_sharpe = -float("inf")
        best_params = None

        # Iterate over the same grid as in hyperscan.py
        for params in generate_param_grid():
            # Create a shallow copy of config and inject overrides for this symbol
            cfg_train = dict(cfg)
            sym_cfg = cfg_train["symbols"][symbol]
            for key, val in params.items():
                sym_cfg[f"{key}_override"] = val

            # Backtest on train CSV
            bt_train = Backtester(
                symbol=symbol,
                config=cfg_train,
                data_dir=temp_dir,
                strategy_name=sym_cfg["strategy"]
            )
            metrics_train = bt_train.run()
            if metrics_train["sharpe"] > best_sharpe:
                best_sharpe = metrics_train["sharpe"]
                best_params = params

        print(f"→ Best train params: {best_params} with Sharpe={best_sharpe:.2f}")

        # 5) Backtest on test set with best_params
        cfg_test = dict(cfg)
        sym_cfg_test = cfg_test["symbols"][symbol]
        for key, val in best_params.items():
            sym_cfg_test[f"{key}_override"] = val

        print(f"=== Testing {symbol} from {test_start.date()} to {test_end.date()} ===")
        bt_test = Backtester(
            symbol=symbol,
            config=cfg_test,
            data_dir=temp_dir,
            strategy_name=sym_cfg_test["strategy"]
        )
        metrics_test = bt_test.run()
        print(f"→ Test metrics: {metrics_test}")

        # 6) Record results
        results.append({
            "symbol": symbol,
            "train_start": cur_train_start.date(),
            "train_end": train_end.date(),
            "test_start": test_start.date(),
            "test_end": test_end.date(),
            **best_params,
            "train_sharpe": best_sharpe,
            "test_sharpe": metrics_test["sharpe"],
            "test_return": metrics_test["total_return"],
            "test_max_drawdown": metrics_test["max_drawdown"],
            "test_win_rate": metrics_test["win_rate"],
            "test_n_trades": metrics_test["n_trades"]
        })

        # Advance window
        cur_train_start = test_start

    # 7) Save all results to output CSV
    out_dir = os.path.dirname(output_csv)
    ensure_dir(out_dir)
    df_res = pd.DataFrame(results)
    df_res.to_csv(output_csv, index=False)
    print(f"\nWalk-forward complete for {symbol}. Results saved to {output_csv}")


if __name__ == "__main__":
    # Example usage: walk-forward on all enabled symbols
    cfg = load_config("config/config.yaml")
    data_dir = "data/klines"
    temp_dir = "data/temp"
    out_dir = "backtesting"
    ensure_dir(out_dir)

    for symbol, sym_cfg in cfg["symbols"].items():
        if not sym_cfg.get("enabled", False):
            continue
        strategy = sym_cfg.get("strategy", "").lower()
        if strategy not in ["mean_reversion", "grid", "ml"]:
            continue

        output_csv = os.path.join(out_dir, f"walkforward_{symbol}.csv")
        walk_forward(
            symbol=symbol,
            cfg=cfg,
            data_dir=data_dir,
            temp_dir=temp_dir,
            output_csv=output_csv,
            train_months=1,
            test_months=1
        )
