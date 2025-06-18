# TRD_BOT_V3/src/ml/train.py

import os
import sys

# Ensure src/ is on sys.path for imports
THIS_DIR = os.path.dirname(__file__)                # .../src/ml
SRC_DIR  = os.path.abspath(os.path.join(THIS_DIR, ".."))  # .../src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from client.futures_client import FuturesClient
from ml.data_loader import load_features_and_trade_labels
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Train ML model from trade history + OHLC.")
    parser.add_argument("--symbol",           type=str,   required=True,
                        help="Trading pair, e.g. XRPUSDT")
    parser.add_argument("--interval",         type=str,   default="1h",
                        help="OHLC interval, e.g. 1h")
    parser.add_argument("--data_dir",         type=str,   default="data/klines",
                        help="Folder with OHLC CSVs")
    parser.add_argument("--lookback",         type=int,   default=50,
                        help="Feature lookback window")
    parser.add_argument("--test_size",        type=float, default=0.2,
                        help="Test split fraction")
    parser.add_argument("--model_dir",        type=str,   default="models",
                        help="Where to save the trained model")
    parser.add_argument("--n_estimators",     type=int,   default=100,
                        help="RandomForest n_estimators")
    parser.add_argument("--max_depth",        type=int,   default=5,
                        help="RandomForest max_depth")
    parser.add_argument("--refresh_interval", type=int,   default=3600,
                        help="Seconds before trade history cache refresh")
    return parser.parse_args()

def load_config(path="config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()

    # 0) Load full bot config (for API credentials & symbols settings)
    cfg = load_config()

    # 1) Instantiate FuturesClient (will pick live/testnet keys)
    client = FuturesClient(cfg)

    # 2) Load features & labels
    X, y = load_features_and_trade_labels(
        symbol=args.symbol,
        interval=args.interval,
        data_dir=args.data_dir,
        lookback=args.lookback,
        client=client,
        refresh_interval=args.refresh_interval
    )
    print(f"Loaded {len(X)} rows for {args.symbol}. Label distribution:")
    print(y.value_counts().to_dict(), "\n")

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, shuffle=False
    )

    # 4) Initialize & fit classifier
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 5) Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, list(clf.classes_).index(1)]
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}\n")

    # 6) Save model
    os.makedirs(args.model_dir, exist_ok=True)
    fname = f"{args.symbol.lower()}_ml_from_trades.pkl"
    model_path = os.path.join(args.model_dir, fname)
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
