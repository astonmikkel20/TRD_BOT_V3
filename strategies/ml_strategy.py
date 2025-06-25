# TRD_BOT_V3/src/strategies/ml_strategy.py

import logging
from .base_strategy import BaseStrategy

from ml.feature_engineering import engineer_features
from ml.model import MLModel
from utils.trade_history_manager import TradeHistoryManager

class MLStrategy(BaseStrategy):
    def __init__(self, client, cfg, symbol: str, pm=None, notifier=None):
        """
        pm is unused in run_sim(); real run() (live) uses PositionManager.
        """
        super().__init__(client, cfg, symbol, pm, notifier)
        self.symbol = symbol
        self.pm = pm

        # Reconcile existing positions
        self.in_position = False
        if self.pm:
            self.pm.reconcile(self.client)
            self.in_position = self.pm.is_in_position(self.symbol)

        # Symbol config
        sym_cfg = cfg["symbols"][symbol]
        self.allocation_pct = sym_cfg.get("allocation_pct", 0)
        total_cap = cfg.get("capital_usdt", 0)
        self.allocation_usdt = (total_cap * self.allocation_pct) / 100.0

        self.leverage = sym_cfg.get("leverage", cfg["defaults"]["leverage"])
        self.position_mode = sym_cfg.get(
            "position_mode", cfg["defaults"]["position_mode"]
        )
        self.stop_loss_pct = sym_cfg.get(
            "stop_loss_pct", cfg["risk_defaults"]["stop_loss_pct"]
        )
        self.take_profit_pct = sym_cfg.get(
            "take_profit_pct", cfg["risk_defaults"]["take_profit_pct"]
        )
        self.max_position_size_usdt = sym_cfg.get(
            "max_position_size_usdt",
            (cfg["risk_defaults"]["max_position_size_pct"] / 100.0) * total_cap
        )

        # Validate strategy
        self.strategy_name = sym_cfg.get("strategy", "").lower()
        if self.strategy_name != "ml":
            raise ValueError(f"{symbol} strategy is not 'ml' in config.yaml")

        # ML block
        ml_cfg = sym_cfg.get("ml", {})
        self.model_path = ml_cfg.get("model_path", "")
        if not self.model_path:
            raise ValueError("ml.model_path must be set for MLStrategy")

        self.threshold_buy = ml_cfg.get("threshold_buy", 0.5)
        self.threshold_sell = ml_cfg.get("threshold_sell", 0.5)
        self.interval = ml_cfg.get("interval", "1h")
        zone = ml_cfg.get("zone", {})
        self.zone_lower = zone.get("lower", float("-inf"))
        self.zone_upper = zone.get("upper", float("inf"))

        # Load model trained on your past fills
        self.model = MLModel(self.model_path)
        self.lookback = ml_cfg.get("lookback", 50)

        # TradeHistoryManager to keep trade cache up to date
        self.thm = TradeHistoryManager(
            symbol=symbol,
            client=self.client,
            refresh_interval=ml_cfg.get("trade_cache_refresh", 3600)  # seconds
        )

    def _compute_order_size(self, current_price: float) -> float:
        desired_notional = self.allocation_usdt * self.leverage
        max_notional = self.max_position_size_usdt * self.leverage
        if desired_notional > max_notional:
            desired_notional = max_notional
        return desired_notional / current_price

    def run_sim(self) -> dict:
        """
        Backtesting path (uses VirtualClient). Returns:
          {"action":"BUY"/"SELL","price":..., "qty":...} or None.
        """
        current_price = self.client.current_price
        if not (self.zone_lower <= current_price <= self.zone_upper):
            return None

        needed = self.lookback + 1
        df_ohlc = self.client.get_historical_klines(self.symbol, self.interval, needed)
        features = engineer_features(df_ohlc, lookback=self.lookback)
        features_clean = features.dropna()
        if features_clean.empty:
            return None

        X_latest = features_clean.iloc[[-1]]
        prob_buy = float(self.model.predict_proba(X_latest)[0])
        quantity = self._compute_order_size(current_price)
        if quantity <= 0:
            return None

        if prob_buy >= self.threshold_buy and not self.in_position:
            self.in_position = True
            return {"action": "BUY", "price": current_price, "qty": quantity}
        if prob_buy <= self.threshold_sell and self.in_position:
            self.in_position = False
            return {"action": "SELL", "price": current_price, "qty": quantity}
        return None

    def run(self):
        """
        Live/paper-trading path. Steps:
          1) Refresh trade cache if stale
          2) Reconcile in_position via PositionManager
          3) Fetch current mark price
          4) Compute features & prob_buy
          5) Place real LIMIT orders and record them
        """
        # 1) Refresh your trade cache (only if older than `refresh_interval`)
        trades_df = self.thm.get_trade_history()

        # 2) Reconcile & update in_position
        if self.pm:
            self.pm.reconcile(self.client)
            self.in_position = self.pm.is_in_position(self.symbol)

        # 3) Get current mark price
        try:
            ticker = self.client.client.get_mark_price(symbol=self.symbol)
            current_price = float(ticker["markPrice"])
        except Exception as e:
            logging.error(f"MLStrategy: failed to fetch mark price for {self.symbol}: {e}")
            return

        # 4) Zone check
        if not (self.zone_lower <= current_price <= self.zone_upper):
            return

        # 5) Fetch candles & compute features
        needed = self.lookback + 1
        df_ohlc = self.client.get_historical_klines(self.symbol, self.interval, needed)
        features = engineer_features(df_ohlc, lookback=self.lookback)
        features_clean = features.dropna()
        if features_clean.empty:
            return

        X_latest = features_clean.iloc[[-1]]
        prob_buy = float(self.model.predict_proba(X_latest)[0])
        quantity = self._compute_order_size(current_price)
        if quantity <= 0:
            return

        # 6) Entry / exit logic with real orders
        if prob_buy >= self.threshold_buy and not self.in_position:
            order = self.client.place_order(
                symbol=self.symbol,
                side="BUY",
                order_type="LIMIT",
                quantity=quantity,
                price=current_price,
                leverage=self.leverage,
                position_side="LONG" if self.position_mode == "ONE_WAY" else "BOTH"
            )
            self.pm.add_order(self.symbol, order["orderId"], "BUY")
            self.in_position = True
            logging.info(f"ML: Placed LIMIT BUY @ {current_price:.2f}, prob={prob_buy:.2f}")

        elif prob_buy <= self.threshold_sell and self.in_position:
            order = self.client.place_order(
                symbol=self.symbol,
                side="SELL",
                order_type="LIMIT",
                quantity=quantity,
                price=current_price,
                leverage=self.leverage,
                position_side="SHORT" if self.position_mode == "HEDGE" else "LONG"
            )
            self.pm.add_order(self.symbol, order["orderId"], "SELL")
            self.in_position = False
            logging.info(f"ML: Placed LIMIT SELL @ {current_price:.2f}, prob={prob_buy:.2f}")

        else:
            logging.info(f"ML: No action for {self.symbol}, prob={prob_buy:.2f}")
