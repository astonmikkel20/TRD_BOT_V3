# TRD_BOT_V3/src/utils/risk_management.py

import os
import pandas as pd
from typing import Dict, List

class RiskManager:
    """
    • Computes rolling correlations among symbols.
    • Scales down allocations if correlations exceed a threshold.
    • Enforces a maximum net exposure (sum of all allocation_pct).
    """

    def __init__(self, config: Dict, data_dir: str):
        """
        Args:
          config: The full config dict (from config.yaml)
          data_dir: Path where OHLC CSVs are stored, e.g. "data/klines"
        """
        self.cfg = config
        self.data_dir = data_dir
        self.history_cache: Dict[str, pd.DataFrame] = {}

    def _load_price_series(self, symbol: str, interval: str, lookback: int) -> pd.Series:
        """
        Loads the last `lookback` closes from CSV.
        Assumes file named "{symbol}_{interval}.csv" in data_dir, with columns:
          open_time, open, high, low, close, volume
        Returns a pd.Series of the last `lookback` close prices.
        """
        key = f"{symbol}_{interval}"
        if key not in self.history_cache:
            path = os.path.join(self.data_dir, f"{symbol}_{interval}.csv")
            df = pd.read_csv(path, parse_dates=["open_time"])
            # Ensure sorted by time
            df = df.sort_values("open_time").reset_index(drop=True)
            self.history_cache[key] = df
        df = self.history_cache[key]
        return df["close"].iloc[-lookback:].reset_index(drop=True)

    def rolling_correlations(self, symbols: List[str], interval: str, lookback: int) -> pd.DataFrame:
        """
        Compute a correlation matrix of percent returns for the latest `lookback` closes.
        Returns a DataFrame of shape (len(symbols), len(symbols)).
        """
        price_dict = {}
        for sym in symbols:
            series = self._load_price_series(sym, interval, lookback)
            price_dict[sym] = series
        price_df = pd.DataFrame(price_dict)
        # Compute returns then correlation
        returns = price_df.pct_change().dropna()
        corr = returns.corr()
        return corr

    def adjust_allocations(self, symbols: List[str], base_alloc: Dict[str, float],
                           corr_threshold: float = 0.8, reduction_pct: float = 0.5) -> Dict[str, float]:
        """
        Scale down allocations for any highly correlated pairs.

        Args:
          symbols: List of symbol tickers to consider (e.g. ["BTCUSDT","ETHUSDT",...]).
          base_alloc: Dict mapping symbol → allocation_pct (e.g. {"BTCUSDT":10, "ETHUSDT":8}).
          corr_threshold: If corr(sym1,sym2) > this, we reduce each of their allocations.
          reduction_pct: The fraction of combined allocation to remove
                         (e.g. 0.5 → cut combined allocation by 50%).
        Returns:
          A new dict of adjusted allocation_pct.
        """
        # 1) Compute correlation matrix over 100 bars by default
        corr = self.rolling_correlations(symbols, interval="1h", lookback=100)

        adjusted = base_alloc.copy()
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if j <= i:
                    continue
                if corr.loc[sym1, sym2] > corr_threshold:
                    # Combined allocation
                    combined = adjusted.get(sym1, 0) + adjusted.get(sym2, 0)
                    if combined <= 0:
                        continue
                    # Amount to remove (e.g. 50% of combined)
                    removal = combined * reduction_pct
                    # Pro‐rata reduction
                    frac1 = adjusted[sym1] / combined
                    frac2 = adjusted[sym2] / combined
                    adjusted[sym1] -= removal * frac1
                    adjusted[sym2] -= removal * frac2

        return adjusted

    def enforce_notional_cap(self, allocations: Dict[str, float],
                             max_net_pct: float) -> Dict[str, float]:
        """
        Ensure sum(allocation_pct) ≤ max_net_pct. If exceeded, scale all down proportionally.
        Args:
          allocations: Dict symbol → allocation_pct
          max_net_pct: e.g. 50 means “total allocations must not exceed 50% of capital.”
        Returns:
          Possibly scaled Dict of allocation_pct.
        """
        total_pct = sum(allocations.values())
        if total_pct <= max_net_pct:
            return allocations
        scale = max_net_pct / total_pct
        return {sym: pct * scale for sym, pct in allocations.items()}
