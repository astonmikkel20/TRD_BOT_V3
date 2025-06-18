# TRD_BOT_V3/src/utils/trade_history_manager.py

import os
import time
import pandas as pd

class TradeHistoryManager:
    """
    Caches your filled-order history on disk and only refreshes it via
    Binance API when the cache is older than `refresh_interval` seconds.
    """

    def __init__(
        self,
        symbol: str,
        client,
        cache_dir: str = "state",
        cache_filename: str = None,
        refresh_interval: int = 3600
    ):
        """
        Args:
          symbol: e.g. "XRPUSDT"
          client: your FuturesClient instance with .client.futures_account_trades()
          cache_dir: folder to store CSV cache
          cache_filename: override default filename
          refresh_interval: seconds before re-fetching from API
        """
        self.symbol = symbol
        self.client = client
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(
            cache_dir,
            cache_filename or f"{symbol}_trades.csv"
        )
        self.refresh_interval = refresh_interval

    def _is_cache_stale(self) -> bool:
        if not os.path.isfile(self.cache_path):
            return True
        return (time.time() - os.path.getmtime(self.cache_path)) > self.refresh_interval

    def _fetch_all_trades(self) -> pd.DataFrame:
        """
        Fetch full trade history via Binance Futures API.
        Expects client.client.futures_account_trades(symbol).
        """
        records = []
        trades = self.client.client.futures_account_trades(symbol=self.symbol)
        for t in trades:
            qty = float(t["qty"])
            records.append({
                "timestamp": pd.to_datetime(t["time"], unit="ms", utc=True),
                "symbol": t["symbol"],
                "side": "BUY" if qty > 0 else "SELL",
                "price": float(t["price"]),
                "qty": abs(qty)
            })
        df = pd.DataFrame(records)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def get_trade_history(self) -> pd.DataFrame:
        """
        Returns a DataFrame of fills, loading from cache if fresh,
        otherwise fetching and updating the CSV cache.
        """
        if self._is_cache_stale():
            df = self._fetch_all_trades()
            df.to_csv(self.cache_path, index=False)
        else:
            df = pd.read_csv(self.cache_path, parse_dates=["timestamp"])
            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df
