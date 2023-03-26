from typing import Dict

from src.data.api_implementations import TestApi, AlpacaAPI, AlpacaTestApi


class ApiController():
    # Static and consitient across all instances
    stock_specific_instances: Dict[str, StockObserver] = {
        # ticker: instance
    }

    keep_alive = True # When to kill the thread.
    _api_instance = None

    @classmethod
    def set_api_instance(cls):
        cls._api_instance = None

    @classmethod
    def get_instance(cls, ticker):
        if ticker in cls.stock_specific_instances:
            return cls.stock_specific_instances[ticker]
        else:
            specific_api = StockObserver(ticker, cls)
            specific_api.update(cls._api_instance.get_current_price(ticker))
            cls.stock_specific_instances[ticker] = specific_api
            return specific_api

    @classmethod
    def remove_instance(cls, ticker):
        if ticker in cls.stock_specific_instances:
            del cls.stock_specific_instances[ticker]

    @classmethod
    def update_prices(cls, timestamp):
        for ticker in cls.stock_specific_instances:
            api_instance = cls.stock_specific_instances[ticker]
            if not timestamp:
                price = cls._api_instance.get_current_price(ticker)
            else:
                price = cls._api_instance.get_price_at_timestamp(ticker, timestamp)
            api_instance.update(price)

    @classmethod
    def signal_shutdown(cls):
        for ticker in cls.stock_specific_instances:
            api_instance = cls.stock_specific_instances[ticker]
            api_instance.update(price)

    @classmethod
    def sell(cls, stock_observer, amount):
        return cls._api_instance.sell(stock_observer.ticker, stock_observer.price, amount)

    @classmethod
    def buy(cls, stock_observer, amount):
        return cls._api_instance.buy(stock_observer.ticker, stock_observer.price, amount)


class StockObserver:

    def __init__(self, ticker, controller):
        self.ticker = ticker
        self.price = None
        self.price_sold = None
        self.price_bought = None
        self.stock_id = None
        self.controller = controller # Static apiController

    def buy(self, amount):
        print(f"Buying {self.ticker} @ {self.price}")
        self.price_bought = self.controller.buy(self, amount)

    def update(self, price):
        self.price = price

    def sell(self, amount):
        print(f"Selling {self.ticker} @ {self.price}")
        self.price_sold = self.controller.sell(self, amount)

    def set_stock_id(self, stock_id):
        self.stock_id = stock_id


# Specific controllers go here

class TestController(ApiController):

    @classmethod
    def set_api_instance(cls):
        cls._api_instance = TestApi()


class AlpacaApiController(ApiController):

    @classmethod
    def set_api_instance(cls):
        cls._api_instance = AlpacaAPI()


class AlpacaTestController(ApiController):

    @classmethod
    def set_api_instance(cls):
        cls._api_instance = AlpacaTestApi()

