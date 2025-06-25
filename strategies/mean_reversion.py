# TRD_BOT_V3/src/strategies/mean_reversion.py

import logging
import pandas as pd
from .base_strategy import BaseStrategy
# Note: we no longer import PositionManager here since run_sim ignores pm
# PositionManager is only used in run() for live trading

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, client, cfg, symbol: str, pm=None, notifier=None):
        """
        pm is unused in run_sim(); real run() (live) uses PositionManager.
        """
        super().__init__(client, cfg, symbol, pm, notifier)
        self.symbol = symbol
        self.pm = pm  # may be None in backtest

        # For live trading, we would reconcile pm here. Backtest ignores pm.
        self.in_position = False

        # Total capital
        total_cap = cfg.get("capital_usdt", 0)

        # Symbol config
        sym_cfg = cfg["symbols"][symbol]
        self.allocation_pct = sym_cfg.get("allocation_pct", 0)
        self.allocation_usdt = (total_cap * self.allocation_pct) / 100.0

        # Leverage & position mode
        self.leverage = sym_cfg.get("leverage", cfg["defaults"]["leverage"])
        self.position_mode = sym_cfg.get("position_mode", cfg["defaults"]["position_mode"])

        # Risk
        self.stop_loss_pct = sym_cfg.get("stop_loss_pct", cfg["risk_defaults"]["stop_loss_pct"])
        self.take_profit_pct = sym_cfg.get("take_profit_pct", cfg["risk_defaults"]["take_profit_pct"])
        self.max_position_size_usdt = sym_cfg.get(
            "max_position_size_usdt",
            (cfg["risk_defaults"]["max_position_size_pct"] / 100.0) * total_cap
        )

        # Validate strategy name
        self.strategy_name = sym_cfg.get("strategy", "").lower()
        if self.strategy_name != "mean_reversion":
            raise ValueError(f"{symbol} strategy is not 'mean_reversion' in config.yaml")

        # Engine defaults or overrides
        mr_cfg = sym_cfg.get("mean_reversion", {})

        self.lookback = sym_cfg.get("lookback_override", mr_cfg.get("lookback", 50))
        self.rsi_period = sym_cfg.get("rsi_period_override", mr_cfg.get("rsi_period", 14))
        self.rsi_oversold = sym_cfg.get("rsi_oversold_override", mr_cfg.get("rsi_oversold", 30))
        self.rsi_overbought = sym_cfg.get("rsi_overbought_override", mr_cfg.get("rsi_overbought", 70))
        self.interval = mr_cfg.get("interval", "1h")
        self.trend_lookback = sym_cfg.get("trend_lookback_override", mr_cfg.get("trend_lookback", 50))
        self.lt_vol_lookback = sym_cfg.get("lt_vol_lookback_override", mr_cfg.get("lt_vol_lookback", 200))
        self.sigma_bank = sym_cfg.get("sigma_bank_override", mr_cfg.get("sigma_bank", [1.5, 2.0, 2.5, 3.0]))

    def _compute_order_size(self, current_price: float) -> float:
        desired_notional = self.allocation_usdt * self.leverage
        max_notional = self.max_position_size_usdt * self.leverage
        if desired_notional > max_notional:
            desired_notional = max_notional
        return desired_notional / current_price

    def _fetch_ohlc(self, interval: str, limit: int) -> pd.DataFrame:
        return self.client.get_historical_klines(self.symbol, interval, limit)

    def _compute_atr(self, df: pd.DataFrame, lookback: int) -> float:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=lookback, min_periods=lookback).mean().iloc[-1]
        return float(atr)

    def _compute_sma(self, series: pd.Series, period: int) -> float:
        return float(series.rolling(window=period, min_periods=period).mean().iloc[-1])

    def _compute_rsi(self, series: pd.Series, period: int) -> float:
        delta = series.diff().dropna()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean().iloc[-1]
        avg_loss = loss.rolling(window=period, min_periods=period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    def run_sim(self) -> dict:
        """
        Simulated “run” for backtesting. 
        Returns a dict {"action": "BUY"/"SELL", "price": float, "qty": float} or None.
        """
        # 1) Get current price from VirtualClient
        current_price = self.client.current_price

        # 2) Compute quantity
        quantity = self._compute_order_size(current_price)
        if quantity <= 0:
            return None

        # 3) Fetch enough candles for calculations
        needed = max(self.lookback + 1, self.trend_lookback + 1, self.rsi_period + 1, self.lt_vol_lookback + 1)
        df = self._fetch_ohlc(self.interval, needed)
        closes = df["close"]

        # 4) Compute Long/Short ATRs
        atr_lt = self._compute_atr(df, self.lt_vol_lookback)
        vol_lt_pct = atr_lt / current_price

        df_st = df.iloc[-(self.lookback + 1):]
        atr_st = self._compute_atr(df_st, self.lookback)
        vol_st_pct = atr_st / current_price

        # 5) Determine σ‐multiplier
        ratio = (vol_st_pct / vol_lt_pct) if vol_lt_pct > 0 else 1.0
        if ratio < 0.8:
            std_mul = self.sigma_bank[0]
        elif ratio < 1.2:
            std_mul = self.sigma_bank[1]
        elif ratio < 1.6:
            std_mul = self.sigma_bank[2]
        else:
            std_mul = self.sigma_bank[3]

        # 6) Compute mean & std for bands
        ma = closes.iloc[-(self.lookback + 1):-1].mean()
        std = closes.iloc[-(self.lookback + 1):-1].std()
        upper_band = ma + std_mul * std
        lower_band = ma - std_mul * std

        # 7) Compute SMA (trend) and RSI
        sma = self._compute_sma(closes, self.trend_lookback)
        rsi = self._compute_rsi(closes, self.rsi_period)

        # 8) Entry / exit logic
        entry_price = current_price
        buy_limit = entry_price
        sell_limit = entry_price

        # BUY if below lower_band, above SMA, RSI oversold, not in position
        if (current_price < lower_band
            and current_price > sma
            and rsi < self.rsi_oversold
            and not self.in_position):
            self.in_position = True
            return {"action": "BUY", "price": buy_limit, "qty": quantity}

        # SELL if above upper_band, RSI overbought, currently in position
        elif (current_price > upper_band
              and rsi > self.rsi_overbought
              and self.in_position):
            self.in_position = False
            return {"action": "SELL", "price": sell_limit, "qty": quantity}

        return None  # no action

    def run(self):
        """
        Live‐trading run (unchanged from Step 3). Not used by backtester.
        """
        raise NotImplementedError("Use run_sim() when backtesting.")


