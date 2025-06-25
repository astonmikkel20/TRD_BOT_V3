# TRD_BOT_V3/src/utils/position_manager.py

import json
import os
from typing import Dict

class PositionManager:
    """
    Tracks open orders and positions via a JSON on disk.
    On startup or periodically, calls Binance to reconcile
    which orders filled and which positions remain.
    """
    def __init__(self, filepath: str = "state/positions.json"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if os.path.isfile(self.filepath):
            with open(self.filepath, "r") as f:
                self.state: Dict[str, Dict] = json.load(f)
        else:
            self.state = {}
            self._save()

    def _save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.state, f, indent=2)

    def add_order(self, symbol: str, order_id: str, side: str):
        """
        Record a newly placed order for `symbol`.
        side = "BUY" or "SELL". Status starts as "OPEN".
        """
        self.state[symbol] = {
            "order_id": str(order_id),
            "side": side,
            "status": "OPEN"
        }
        self._save()

    def mark_filled(self, symbol: str):
        """Mark the recorded order for `symbol` as FILLED."""
        if symbol in self.state:
            self.state[symbol]["status"] = "FILLED"
            self._save()

    def clear(self, symbol: str):
        """Remove any record for `symbol` (e.g., order canceled or position closed)."""
        if symbol in self.state:
            del self.state[symbol]
            self._save()

    def reconcile(self, client):
        """
        Check each recorded order against Binance:
         1) If still in open orders → leave as is.
         2) If not open, but there's a nonzero position on Binance → mark FILLED.
         3) If not open and no position → clear record.
        Assumes:
          - client.get_open_orders(symbol) → list of orders
          - client.get_account_positions() → list of positions like {"symbol": "...", "positionAmt": "..."}
        """
        for symbol, record in list(self.state.items()):
            order_id = record["order_id"]

            # 1) Check if order is still open
            try:
                open_orders = client.get_open_orders(symbol)
            except Exception:
                open_orders = []

            still_open = any(str(o["orderId"]) == order_id for o in open_orders)
            if still_open:
                continue

            # 2) Order no longer open → check if a position remains
            try:
                positions = client.get_account_positions()
            except Exception:
                positions = []

            pos = next((p for p in positions if p["symbol"] == symbol), None)
            if pos and float(pos.get("positionAmt", 0)) != 0:
                # A position remains → mark as FILLED
                self.state[symbol]["status"] = "FILLED"
                self._save()
            else:
                # No order and no position → remove record
                del self.state[symbol]
                self._save()

    def is_in_position(self, symbol: str) -> bool:
        """Return True if `symbol` has status == 'FILLED' in state."""
        rec = self.state.get(symbol)
        return rec is not None and rec.get("status") == "FILLED"
