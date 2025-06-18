# scripts/test_data_loader.py
import pandas as pd
from ml.data_loader import load_features_and_trade_labels

class DummyClient:
    pass  # no methods needed because our loader no longer uses client for features

X, y = load_features_and_trade_labels(
    symbol="SOLUSDT",
    interval="1h",
    data_dir="data/klines",
    lookback=3,
    client=DummyClient(),
    refresh_interval=3600
)
print("Features shape:", X.shape)
print("Labels distribution:", y.value_counts().to_dict())
