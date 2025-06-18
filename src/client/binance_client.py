# TRD_BOT_V3/src/client/binance_client.py

import logging
import time
import pandas as pd
from binance.client import Client as BinancePyClient
from binance.exceptions import BinanceAPIException

class BinanceClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Wrapper sobre python-binance.Client. Si testnet=True, apunta a Testnet.
        """
        if testnet:
            # El constructor nativo de python-binance diferencia testnet mediante un flag
            self.client = BinancePyClient(api_key, api_secret, testnet=True)
            # Redefinimos la URL para Testnet
            self.client.API_URL = "https://testnet.binance.vision/api"
            logging.info("BinanceClient inicializado en TESTNET")
        else:
            self.client = BinancePyClient(api_key, api_secret)
            logging.info("BinanceClient inicializado en LIVE")

        # Pausa breve para respetar rate limits
        self._sleep_interval = 0.5

    def _sleep(self):
        """ Pequeña espera tras cada llamada para no saturar rate limit. """
        time.sleep(self._sleep_interval)

    def get_historical_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """
        Descarga hasta <limit> velas (klines) del par <symbol> con intervalo <interval>.
        Devuelve un DataFrame con columnas:
          [open_time, open, high, low, close, volume, close_time, quote_asset_vol, num_trades, taker_buy_base_vol, taker_buy_quote_vol]
        """
        try:
            raw = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            self._sleep()

            df = pd.DataFrame(raw, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_vol", "num_trades",
                "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
            ])
            # Convertir timestamps y tipos numéricos
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            for col in ["open", "high", "low", "close", "volume", "quote_asset_vol", "taker_buy_base_vol", "taker_buy_quote_vol"]:
                df[col] = df[col].astype(float)

            return df
        except BinanceAPIException as e:
            logging.error(f"[get_historical_klines] BinanceAPIException: {e}")
            raise
        except Exception as e:
            logging.error(f"[get_historical_klines] Error inesperado: {e}")
            raise

    def get_balance(self, asset: str) -> float:
        """
        Devuelve el balance “free” del asset (p. ej. "BTC" o "USDT").
        """
        try:
            bal = self.client.get_asset_balance(asset=asset)
            self._sleep()
            return float(bal["free"])
        except BinanceAPIException as e:
            logging.error(f"[get_balance] BinanceAPIException para {asset}: {e}")
            return 0.0
        except Exception as e:
            logging.error(f"[get_balance] Error inesperado para {asset}: {e}")
            return 0.0

    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None) -> dict:
        """
        Coloca una orden LIMIT o MARKET en el par <symbol>.
          - side: "BUY" o "SELL"
          - order_type: "LIMIT" o "MARKET"
          - Si es LIMIT, se debe pasar price (float).
        Retorna el dict de la respuesta de Binance o None si falla.
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity
        }
        if order_type == "LIMIT":
            if price is None:
                raise ValueError("Para orden LIMIT se requiere un precio.")
            params["price"] = str(price)
            params["timeInForce"] = "GTC"

        try:
            order = self.client.create_order(**params)
            logging.info(f"[place_order] {order_type} {side} en {symbol}, qty={quantity}, price={price} -> orderId={order['orderId']}")
            self._sleep()
            return order
        except BinanceAPIException as e:
            logging.error(f"[place_order] BinanceAPIException: {e}")
            return None
        except Exception as e:
            logging.error(f"[place_order] Error inesperado: {e}")
            return None

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        """
        Cancela la orden con ID <order_id> en <symbol>. 
        Retorna el dict de la respuesta o {} si falla.
        """
        try:
            result = self.client.cancel_order(symbol=symbol, orderId=order_id)
            logging.info(f"[cancel_order] Orden {order_id} cancelada en {symbol}")
            self._sleep()
            return result
        except BinanceAPIException as e:
            logging.error(f"[cancel_order] BinanceAPIException: {e}")
            return {}
        except Exception as e:
            logging.error(f"[cancel_order] Error inesperado: {e}")
            return {}

    def get_open_orders(self, symbol: str) -> list:
        """
        Devuelve la lista de órdenes abiertas (no ejecutadas) para <symbol>.
        """
        try:
            orders = self.client.get_open_orders(symbol=symbol)
            self._sleep()
            return orders
        except BinanceAPIException as e:
            logging.error(f"[get_open_orders] BinanceAPIException: {e}")
            return []
        except Exception as e:
            logging.error(f"[get_open_orders] Error inesperado: {e}")
            return []
