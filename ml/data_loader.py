import os, glob, pandas as pd
from utils.trade_history_manager import TradeHistoryManager

def load_features_and_trade_labels(
    symbol: str,
    interval: str,
    data_dir: str,
    lookback: int,
    client,
    refresh_interval: int = 3600
):
    # 1) Find the OHLC CSV via glob: SYMBOL_*_INTERVAL.csv
    pattern = os.path.join(data_dir, f"{symbol}_*_{interval}.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No OHLC CSV matching: {pattern}")
    df = pd.read_csv(files[0], parse_dates=["open_time"])

    # 2) Fetch trades via TradeHistoryManager
    thm = TradeHistoryManager(symbol, client, cache_dir="state", refresh_interval=refresh_interval)
    trades_df = thm.get_trade_history()

    # 3) Build a simple features DataFrame (returns + moving average)
    df_feat = pd.DataFrame({
        "close": df["close"],
        "return": df["close"].pct_change(),
        "ma":    df["close"].rolling(window=lookback).mean()
    })

    # 4) Label: next bar up/down
    labels = (df["close"].shift(-1) > df["close"]).astype(int)

    # 5) Drop lookback rows + last NaN
    X = df_feat.iloc[lookback:-1].reset_index(drop=True)
    y = labels.iloc[lookback:-1].reset_index(drop=True)

    return X, y
