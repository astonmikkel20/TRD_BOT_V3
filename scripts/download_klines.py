# scripts/download_klines.py

import os
import yaml
import warnings
import pandas as pd
from binance.client import Client

# ──────────────────────────────────────────────────────────────────────────────
# Suppress that pandas/Binance deprecation warning about date parsing
warnings.filterwarnings(
    "ignore",
    message="Parsing dates involving a day of month without a year",
    category=DeprecationWarning
)
# ──────────────────────────────────────────────────────────────────────────────

# CONFIGURATION
SYMBOLS      = ["ETHUSDT", "BTCUSDT"]
INTERVAL     = Client.KLINE_INTERVAL_1HOUR
START_DATE   = "1 Jan, 2021"
END_DATE     = "1 Jan, 2025"
OUT_DIR      = os.path.join("data", "klines")
SECRETS_PATH = os.path.join("config", "secrets.yaml")
CONFIG_PATH  = os.path.join("config", "config.yaml")
# ──────────────────────────────────────────────────────────────────────────────

def load_binance_credentials(path):
    with open(path, "r") as f:
        sec = yaml.safe_load(f) or {}
    if isinstance(sec.get("binance"), dict):
        bc = sec["binance"]
        api_key    = bc.get("api_key", "")
        api_secret = bc.get("api_secret", "")
    else:
        api_key    = sec.get("api_key", "")
        api_secret = sec.get("api_secret", "")
    if not api_key or not api_secret:
        raise ValueError("Missing api_key & api_secret in secrets.yaml")
    return api_key, api_secret

def ensure_out_dir(path):
    os.makedirs(path, exist_ok=True)

def interval_suffix(interval):
    if interval == Client.KLINE_INTERVAL_1HOUR: return "1h"
    if interval == Client.KLINE_INTERVAL_4HOUR: return "4h"
    if interval == Client.KLINE_INTERVAL_1DAY:  return "1d"
    return "".join(ch for ch in interval if ch.isalnum()).lower()

def download_klines(client, symbol, interval, start, end, out_dir, contract_type):
    print(f"Downloading {symbol} {contract_type} {interval} from {start} to {end}...")
    klines = client.get_historical_klines(symbol, interval, start, end)

    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "_1","_2","_3","_4","_5","_6"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df[["open_time","open","high","low","close","volume"]]
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    suf = interval_suffix(interval)
    filename = f"{symbol}_{contract_type}_{suf}.csv"
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    print(f"→ Saved {len(df)} rows to {path}\n")

def main():
    # Load config to get each symbol’s contract_type
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    api_key, api_secret = load_binance_credentials(SECRETS_PATH)
    client = Client(api_key, api_secret)
    ensure_out_dir(OUT_DIR)

    for symbol in SYMBOLS:
        sym_cfg = cfg["symbols"].get(symbol, {})
        contract_type = sym_cfg.get("contract_type", "PERPETUAL")
        download_klines(
            client, symbol, INTERVAL,
            START_DATE, END_DATE,
            OUT_DIR, contract_type
        )

if __name__ == "__main__":
    main()
