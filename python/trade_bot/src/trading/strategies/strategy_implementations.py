from enum import Enum
from abc import abstractmethod
from time import sleep
from typing import Dict, Any

from src.common.enums import PositionState, EnumFromStrMixin
from src.data.api_controllers import StockObserver


class Strategy:

    def __init__(self, stock: StockObserver, money_to_spend: float, strategy_specific_args: Dict) -> None:
        self.stock = stock # Stock observer instance
        self.state = PositionState.OPEN
        self.money_to_spend = money_to_spend
        self.shares_owned = 0
        self.other_args = strategy_specific_args
        self.trade_id = None

    def update_state(self, state: PositionState) -> None:
        print(f"{self.stock.ticker} has state {self.state}")
        self.state = state

    def set_shares(self, shares: int) -> None:
        self.shares_owned = shares

    def __getattr__(self, item: str) -> Any:
        if item in self.other_args:
            return self.other_args[item]
        else:
            raise ValueError(f"{self.__class__.__name__} has no attribute {item}")

    @abstractmethod
    def execute(self) -> None:
        pass

    @staticmethod
    def get_type_from_str(strat_name: str):
        return Strategies.from_str(strat_name)

    # def wait_until_update(self):
    #     sleep(UPDATE_WAIT)

    def open(self) -> None:
        self.enter()
        self.post_entry()

    def close(self) -> None:
        self.exit()
        self.post_exit()

    @abstractmethod
    def enter(self) -> None:
        pass

    @abstractmethod
    def exit(self) -> None:
        pass

    def post_entry(self) -> None:
        self.shares_owned = self.money_to_spend / self.stock.price_bought
        self.update_state(PositionState.WAIT_SELL)

    def post_exit(self) -> None:
        self.shares_owned = 0
        self.update_state(PositionState.CLOSED)


class SimpleEnterAndExitLong(Strategy):

    def execute(self):
        expected_exit = self.exit_gain * self.stock.price
        expected_entry = self.entry_gain * self.stock.price

        while self.stock.price > expected_entry:
            self.wait_until_update()

        self.open()
        while self.stock.price < expected_exit:
            self.wait_until_update()

        self.close()

    def enter(self):
        self.stock.buy(round(self.money_to_spend / self.stock.price))

    def exit(self):
        self.stock.sell(self.shares_owned)

    def get_name(self):
        return f"{self.__class__.__name__} {self.stock.ticker}"


class Strategies(EnumFromStrMixin, Enum):
    SIMPLE_ENTER_EXIT_LONG = SimpleEnterAndExitLong

    def get_strategy_from_str(self, strategy_name):
        return self.from_str(strategy_name).value