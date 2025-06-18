# TRD_BOT_V3/src/ml/feature_engineering.py

import pandas as pd
import numpy as np

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the RSI (Relative Strength Index) over `period` bars.
    Returns a Series of same length, with NaN for the first `period` bars.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss.replace(0, 1e-8))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute ATR (Average True Range) over `period` bars.
    Expects df with columns ['high','low','close'].
    Returns a Series of ATR values.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr

def engineer_features(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    Create a DataFrame of features for each bar in df:
      • RSI(14)
      • SMA(close, lookback)
      • EMA(close, lookback/2)
      • ATR(14)
      • Momentum: close / close.shift(lookback) - 1
      • Volume change: volume / volume.shift(lookback) - 1
    Returns a DataFrame of shape (len(df), n_features), with NaNs for early rows.
    """
    features = pd.DataFrame(index=df.index)

    close = df["close"]
    volume = df["volume"]

    # RSI(14)
    features["rsi_14"] = compute_rsi(close, period=14)

    # SMA(close, lookback)
    features[f"sma_{lookback}"] = compute_sma(close, period=lookback)

    # EMA(close, lookback/2)
    half_lb = max(2, lookback // 2)
    features[f"ema_{half_lb}"] = compute_ema(close, period=half_lb)

    # ATR(14)
    features["atr_14"] = compute_atr(df, period=14)

    # Momentum (return over lookback)
    features[f"mom_{lookback}"] = close / close.shift(lookback) - 1

    # Volume change over lookback
    features[f"vol_chg_{lookback}"] = volume / volume.shift(lookback) - 1

    return features
