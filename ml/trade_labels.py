# TRD_BOT_V3/src/ml/trade_labels.py

import pandas as pd

def create_bar_labels_from_trades(
    trades_df: pd.DataFrame,
    ohlc: pd.DataFrame,
    bar_interval: str = "1h"
) -> pd.Series:
    """
    trades_df: DataFrame with at least ['timestamp','symbol','side','price','qty'] for ONE symbol.
    ohlc: DataFrame of candle data with 'open_time' as datetime.
    Returns a Series indexed by ohlc['open_time'], labeled:
      1 if last fill in that bar was BUY
      0 if last fill was SELL
      NaN if no fill that bar
    """
    bars = ohlc[["open_time"]].copy()
    bars.set_index("open_time", inplace=True)
    bars = bars.sort_index()

    trades = trades_df.copy()
    trades["bar_time"] = trades["timestamp"].dt.floor(bar_interval)
    last_side_per_bar = trades.groupby("bar_time")["side"].last()
    labels = bars.index.to_series().map(last_side_per_bar)
    labels_num = labels.map({"BUY": 1, "SELL": 0})
    return labels_num

# The previous function load_trade_history() can be removed,
# since fetching is now handled by TradeHistoryManager.
