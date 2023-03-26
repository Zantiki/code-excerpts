from typing import Callable

from src.data.api_controllers import ApiController


class Position:
    """
    The individual position, that has a strategy and associated actions
    """

    def __init__(self, ticker: str, strategy: Callable, money_to_spend: float, strategy_args: dict, controller: ApiController) -> None:
        self.controller = controller # Static api-controller type
        self.stock = controller.get_instance(ticker)
        self.currency = "USD"
        self.current_strategy = strategy(self.stock, money_to_spend, strategy_args)
        self.trade_id = None

    def set_trade_id(self, id: int) -> None:
        self.trade_id = id
        self.current_strategy.trade_id = id

    def do_strategy(self) -> None:
        self.current_strategy.execute()

    def get_entry(self) -> float:
        return self.stock.price_bought

    def get_exit(self) -> float:
        return self.stock.price_sold

    def get_shares_owned(self) -> int:
        if not self.current_strategy.shares_owned:
            return round(self.current_strategy.money_to_spend / self.get_entry())
        return self.current_strategy.shares_owned

    def get_recent_total_value(self) -> float:
        return self.stock.price * self.get_shares_owned()


