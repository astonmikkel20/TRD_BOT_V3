import os
import sys

# 1) Ensure src/ is on sys.path so we can import strategies
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import pandas as pd
import numpy as np
from importlib import import_module

class Backtester:
    """
    Replays historical OHLC data and simulates strategy logic via run_sim().
    Outputs performance metrics.
    """

    def __init__(self, symbol: str, config: dict, data_dir: str, strategy_name: str):
        self.symbol = symbol
        self.cfg = config
        self.data_dir = data_dir
        self.strategy_name = strategy_name

        # Build filename from symbol, contract_type, interval
        sym_cfg = config["symbols"][symbol]
        contract_type = sym_cfg.get("contract_type", "PERPETUAL")
        interval = (
            sym_cfg.get("mean_reversion", {}).get("interval")
            or sym_cfg.get("grid", {}).get("interval")
            or sym_cfg.get("ml", {}).get("interval")
            or "1h"
        )
        filename = f"{symbol}_{contract_type}_{interval}.csv"
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Missing historical CSV: {filepath}")

        # Load data
        self.df = pd.read_csv(filepath, parse_dates=["open_time"])
        self.df.sort_values("open_time", inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Import the correct strategy module
        if strategy_name == "grid":
            module_path = "strategies.grid_strategy"
        elif strategy_name == "mean_reversion":
            module_path = "strategies.mean_reversion"
        elif strategy_name == "ml":
            module_path = "strategies.ml_strategy"
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        mod = import_module(module_path)
        if strategy_name == "ml":
            class_name = "MLStrategy"
        else:
            class_name = "".join(p.capitalize() for p in strategy_name.split("_")) + "Strategy"

        self.StrategyClass = getattr(mod, class_name)

        # Placeholders
        self.equity_curve = []
        self.trades = []

    def run(self) -> dict:
        client = VirtualClient(self.df)
        strat = self.StrategyClass(client, self.cfg, self.symbol, pm=None)

        # Apply overrides
        sym_cfg = self.cfg["symbols"][self.symbol]
        for key in [
            "lookback_override", "std_dev_multiplier_override",
            "rsi_period_override", "rsi_oversold_override",
            "rsi_overbought_override", "trend_lookback_override"
        ]:
            if key in sym_cfg:
                setattr(strat, key.replace("_override", ""), sym_cfg[key])

        initial_equity = self.cfg.get("capital_usdt", 100000)
        cash = float(initial_equity)
        position = 0.0
        entry_price = 0.0

        for idx, row in self.df.iterrows():
            client.current_index = idx
            client.current_price = float(row["close"])
            signal = strat.run_sim()
            if signal:
                act = signal["action"]
                price = signal["price"]
                qty = signal["qty"]
                if act == "BUY" and position == 0.0:
                    position = qty
                    entry_price = price
                    cash -= qty * price
                    self.trades.append({
                        "timestamp": row["open_time"],
                        "type": "BUY",
                        "price": entry_price,
                        "qty": position
                    })
                elif act == "SELL" and position > 0.0:
                    proceeds = position * price
                    cash += proceeds
                    pnl = (price - entry_price) * position
                    self.trades[-1].update({
                        "exit_time": row["open_time"],
                        "exit_price": price,
                        "pnl": pnl
                    })
                    position = 0.0
                    entry_price = 0.0

            mtm = position * client.current_price
            self.equity_curve.append(cash + mtm)

        # Metrics
        eq = np.array(self.equity_curve)
        returns = pd.Series(eq).pct_change().dropna()
        total_return = (eq[-1] / initial_equity) - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(365 * 24) if returns.std() else 0.0
        max_dd = self._max_drawdown(eq)
        wins = [t for t in self.trades if t.get("pnl", 0) > 0]
        win_rate = len(wins) / max(1, len(self.trades))

        return {
            "total_return": float(total_return),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
            "n_trades": len(self.trades)
        }

    @staticmethod
    def _max_drawdown(arr: np.ndarray) -> float:
        peak = np.maximum.accumulate(arr)
        dd = (arr - peak) / peak
        return float(dd.min())

class VirtualClient:
    """
    Provides the same interface that your strategies expect, using historical data.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.current_index = -1
        self.current_price = None

    def get_mark_price(self, symbol: str):
        return {"markPrice": str(self.current_price)}

    def place_order(self, symbol: str, side: str, order_type: str,
                    quantity: float, price: float, leverage: int, position_side: str):
        return {"orderId": -1, "status": "FILLED"}

    def get_historical_klines(self, symbol: str, interval: str, lookback: int) -> pd.DataFrame:
        start = max(0, self.current_index - lookback)
        subset = self.df.iloc[start:self.current_index+1].copy()
        return subset[["open_time","open","high","low","close","volume"]].reset_index(drop=True)

    def get_open_orders(self, symbol: str):
        return []

    def get_account_positions(self):
        return []
