# src/strategies/base_strategy.py

class BaseStrategy:
    """
    Common parent class for all strategies.
    - client: Market data & order API (or VirtualClient)
    - cfg: Full bot configuration dict
    - symbol: Trading symbol (e.g. "SOLUSDT")
    - pm: PositionManager instance (or None in backtests)
    - notifier: NotificationManager (optional)
    """

    def __init__(self, client, cfg, symbol: str, pm=None, notifier=None):
        self.client = client
        self.cfg = cfg
        self.symbol = symbol
        self.pm = pm
        self.notifier = notifier

    def run_sim(self):
        """
        Backtest-only method: should be overridden in child classes.
        Return a dict like {"action":"BUY"/"SELL", "price":..., "qty":...} or None.
        """
        raise NotImplementedError("run_sim() must be implemented by strategy subclasses")

    def run(self):
        """
        Live/paper-trading method: overridden by child classes.
        """
        raise NotImplementedError("run() must be implemented by strategy subclasses")
