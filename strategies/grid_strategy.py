# src/strategies/grid_strategy.py

import logging
import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy

class GridStrategy(BaseStrategy):
    """
    Simple volatility‐based grid strategy:
    - Compute ATR-based band around current price
    - Divide into N levels spaced by pct spacing
    - Buy at each level when price dips; sell when price rises
    """

    def __init__(self, client, cfg, symbol: str, pm=None):
        super().__init__(client, cfg, symbol)
        self.pm = pm
        self.symbol = symbol

        # Configuration
        sym_cfg = cfg["symbols"][symbol]
        self.allocation_pct = sym_cfg.get("allocation_pct", 0)
        total_cap = cfg.get("capital_usdt", 0)
        self.allocation_usdt = total_cap * self.allocation_pct / 100.0

        self.leverage = sym_cfg.get("leverage", cfg["defaults"]["leverage"])
        self.position_mode = sym_cfg.get("position_mode", cfg["defaults"]["position_mode"])
        self.stop_loss_pct = sym_cfg.get("stop_loss_pct", cfg["risk_defaults"]["stop_loss_pct"])
        self.take_profit_pct = sym_cfg.get("take_profit_pct", cfg["risk_defaults"]["take_profit_pct"])

        grid_cfg = sym_cfg.get("grid", {})
        self.vol_lookback    = grid_cfg.get("vol_lookback", 20)
        self.vol_multiplier  = grid_cfg.get("vol_multiplier", 2.0)
        self.base_spacing_pct = grid_cfg.get("base_spacing_pct", 0.01)

        # State
        self.in_position = False
        if self.pm:
            self.pm.reconcile(self.client)
            self.in_position = self.pm.is_in_position(self.symbol)

    def _compute_order_size(self, price: float) -> float:
        """Calculate how many contracts to buy/sell."""
        notional = self.allocation_usdt * self.leverage
        return notional / price

    def _determine_grid_parameters(self, df: pd.DataFrame, current_price: float):
        """
        Based on recent volatility, set lower/upper band and grid levels.
        """
        # ATR as proxy: use high-low range
        recent = df["high"].iloc[-self.vol_lookback:] - df["low"].iloc[-self.vol_lookback:]
        vol = recent.mean()
        band = vol * self.vol_multiplier

        lower = current_price - band
        upper = current_price + band

        # number of levels = band width / (spacing_pct * price)
        raw_levels = (upper - lower) / (self.base_spacing_pct * current_price)
        levels = max(1, int(raw_levels))  # at least one level

        spacing = (upper - lower) / levels
        return lower, upper, levels, spacing

    def run_sim(self) -> dict:
        """
        Backtest logic: returns {"action","price","qty"} or None.
        """
        # 1) Load enough bars
        needed = self.vol_lookback + 1
        df = self.client.get_historical_klines(self.symbol, "1h", needed)

        if df is None or len(df) < needed:
            return None

        current_price = self.client.current_price

        # 2) Compute grid
        lower, upper, levels, spacing = self._determine_grid_parameters(df, current_price)
        grid_levels = [lower + i * spacing for i in range(levels + 1)]

        # 3) Tolerance-based matching
        tol = self.base_spacing_pct * current_price
        for level in grid_levels:
            if abs(current_price - level) <= tol:
                qty = self._compute_order_size(current_price)
                if qty <= 0:
                    return None

                # BUY if not in position, otherwise SELL
                if not self.in_position:
                    self.in_position = True
                    return {"action": "BUY", "price": current_price, "qty": qty}
                else:
                    self.in_position = False
                    return {"action": "SELL", "price": current_price, "qty": qty}

        return None

    def run(self):
        """
        Live/paper-trading logic.
        """
        # 1) Reconcile current position
        if self.pm:
            self.pm.reconcile(self.client)
            self.in_position = self.pm.is_in_position(self.symbol)

        # 2) Fetch mark price
        try:
            ticker = self.client.get_mark_price(self.symbol)
            current_price = float(ticker)
        except Exception as e:
            logging.error(f"GridStrategy: failed to fetch mark price: {e}")
            return

        # 3) Fetch historical bars
        needed = self.vol_lookback + 1
        df = self.client.get_historical_klines(self.symbol, "1h", needed)
        if df is None or len(df) < needed:
            return

        # 4) Compute grid and tolerance
        lower, upper, levels, spacing = self._determine_grid_parameters(df, current_price)
        grid_levels = [lower + i * spacing for i in range(levels + 1)]
        tol = self.base_spacing_pct * current_price

        # 5) Check for entry/exit
        for level in grid_levels:
            if abs(current_price - level) <= tol:
                qty = self._compute_order_size(current_price)
                if qty <= 0:
                    return

                side = "BUY" if not self.in_position else "SELL"
                order = self.client.place_order(
                    symbol=self.symbol,
                    side=side,
                    order_type="LIMIT",
                    quantity=qty,
                    price=current_price,
                    timeInForce="GTC",
                    positionSide="LONG" if self.position_mode == "ONE_WAY" else "BOTH"
                )
                if self.pm:
                    self.pm.add_order(self.symbol, order["orderId"], side)
                self.in_position = not self.in_position
                logging.info(f"Grid: {side} {self.symbol} @ {current_price:.4f}")
                break  # only one order per bar
