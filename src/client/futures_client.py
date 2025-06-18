# src/client/futures_client.py

import os
import time
import yaml
from binance.client import Client
from binance.exceptions import BinanceAPIException

class FuturesClient:
    """
    Wrapper around python-binance Client for Binance Futures (USDT-M perpetual).
    Automatically picks live vs. testnet credentials from secrets.yaml,
    and syncs local time offset to prevent timestamp errors.
    """

    def __init__(self, cfg: dict, secrets_path: str = "config/secrets.yaml"):
        # 1) Load API credentials from secrets.yaml
        if not os.path.isfile(secrets_path):
            raise FileNotFoundError(f"Missing secrets file: {secrets_path}")
        with open(secrets_path, "r") as f:
            sec = yaml.safe_load(f) or {}

        bn = sec.get("binance", {})
        live_key    = bn.get("api_key")
        live_secret = bn.get("api_secret")
        test_key    = bn.get("testnet_api_key")
        test_secret = bn.get("testnet_api_secret")

        # 2) Determine whether to use testnet
        use_test = cfg.get("exchange", {}).get("testnet", False)

        if use_test:
            if not (test_key and test_secret):
                raise ValueError("Testnet API credentials missing in secrets.yaml")
            api_key, api_secret = test_key, test_secret
        else:
            if not (live_key and live_secret):
                raise ValueError("Live API credentials missing in secrets.yaml")
            api_key, api_secret = live_key, live_secret

        # 3) Instantiate the python-binance client
        #    Note: in python-binance v1.x, you can pass testnet=True for futures
        self.client = Client(api_key, api_secret, testnet=use_test)

        # 4) Sync time offset to avoid timestamp errors (±1s tolerance)
        try:
            server_time = self.client.get_server_time()
            server_ts = int(server_time["serverTime"])
            local_ts = int(time.time() * 1000)
            self.client.TIME_OFFSET = server_ts - local_ts
        except Exception as e:
            # If time sync fails, log but continue; signed calls may fail
            print(f"[FuturesClient] Warning: failed to sync server time: {e}")

    def get_mark_price(self, symbol: str) -> float:
        """
        Returns the current mark price for a symbol.
        """
        resp = self.client.futures_mark_price(symbol=symbol)
        # futures_mark_price returns a dict or list; ensure dict
        if isinstance(resp, list):
            resp = resp[0]
        return float(resp["markPrice"])

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float = None,
        timeInForce: str = "GTC",
        positionSide: str = None,
        leverage: int = None
    ) -> dict:
        """
        Places a futures order.
        - side: "BUY" or "SELL"
        - order_type: "LIMIT", "MARKET", etc.
        - quantity: contract quantity
        - price: limit price (required for LIMIT orders)
        - timeInForce: "GTC", "IOC", etc.
        - positionSide: "LONG" or "SHORT" (hedge mode)
        - leverage: integer leverage (set before placing)
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity
        }
        if price is not None:
            params["price"] = price
            params["timeInForce"] = timeInForce
        if positionSide:
            params["positionSide"] = positionSide
        try:
            order = self.client.futures_create_order(**params)
            return order
        except BinanceAPIException as e:
            print(f"[FuturesClient] Order error: {e}")
            raise

    def cancel_order(self, symbol: str, orderId: int) -> dict:
        """
        Cancels a futures order by ID.
        """
        return self.client.futures_cancel_order(symbol=symbol, orderId=orderId)

    def get_open_orders(self, symbol: str) -> list:
        """
        Returns a list of open futures orders for the symbol.
        """
        return self.client.futures_get_open_orders(symbol=symbol)

    def get_account_positions(self) -> list:
        """
        Returns current futures position information.
        Each entry contains 'symbol', 'positionAmt', etc.
        """
        return self.client.futures_position_information()
