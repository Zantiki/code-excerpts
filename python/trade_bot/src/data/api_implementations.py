from alpaca_trade_api.common import URL  # type: ignore
import alpaca_trade_api as alpaca  # type: ignore
from time import sleep
from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta
from typing import Dict

class AbstractApi():

    @abstractmethod
    def buy(self, ticker, price, amount):
        pass

    @abstractmethod
    def sell(self, ticker, price, amount):
        pass

    @abstractmethod
    def get_current_price(self, ticker):
        pass

    @abstractmethod
    def get_position(self, ticker):
        pass

    @abstractmethod
    def get_historical(self, ticker, upper_bound, lower_bound, max_points):
        """Must always be pandas"""
        pass

    def get_price_at_timestamp(self, ticker, timestamp):
        last_minute_history = self.get_historical(
            ticker,
            upper_bound=timestamp,
            lower_bound=timestamp - timedelta(minutes=1),
            max_points=60
        )
        return float(last_minute_history[0]["close"])


class TestApi(AbstractApi):
    has_been_called = 0 # Ensure everything after first call is larger than before
    initial_value = 1
    price_flip: Dict[str, bool] = {
        # Ticker: bool to flip
    }

    def get_historical(self, ticker, upper_bound, lower_bound, max_points):
        pass

    def get_position(self, ticker):
        pass

    def get_current_price(self, ticker):
        self.has_been_called += 1
        if ticker not in self.price_flip:
            self.price_flip[ticker] = True
        if self.price_flip[ticker]:
            # print(f"{ticker}: {TestApi.initial_value / self.has_been_called}")
            return TestApi.initial_value / self.has_been_called
        else:
            # print(f"{ticker}: {TestApi.initial_value * self.has_been_called}")
            return TestApi.initial_value * self.has_been_called

    def sell(self, ticker, price, amount):
        return TestApi.initial_value * self.has_been_called # Price it was sold at

    def buy(self, ticker, price, amount):
        self.price_flip[ticker] = False
        return TestApi.initial_value  # Price it was sold at


class AlpacaAPI(AbstractApi):

    ALPACA_ENDPOINT = "https://paper-api.alpaca.markets"

    # 390
    MINUTES_PR_TRADE_DAY = 390

    def __init__(self):
        key_id = get_config_key("alpaca_key_id")
        key = get_config_key("alpaca_key")
        self.api = alpaca.REST(key_id=key_id, secret_key=key, base_url=URL(self.ALPACA_ENDPOINT))

        # Todo: how do i deal with this,
        #  have mutable and immutable positions, i.e ones with rolling wins/losses and others with absolute
        # percentage-factor of buy-price
        self.take_profit = 10
        self.stop_loss = 1

    def get_historical(self, ticker, upper_bound, lower_bound, max_points=None):
        if not max_points:
            max_points = round(timedelta(upper_bound - lower_bound).seconds)

        barset = self.api.get_barset(ticker, 'day',
                                     start=lower_bound.isoformat(), end=upper_bound.isoformat(),
                                     limit=max_points)
        bars = barset[ticker]
        return bars

    def time_to_market_close(self):
        clock = self.api.get_clock()
        return (clock.next_close - clock.timestamp).total_seconds()

    def wait_for_market_open(self):
        clock = self.api.get_clock()
        if not clock.is_open:
            time_to_open = (clock.next_open - clock.timestamp).total_seconds()
            sleep(round(time_to_open))

    def buy(self, ticker, max_price, amount):
        stop_loss = str(round(max_price-(self.stop_loss*max_price), 2))
        take_profit = str(round(max_price + (self.take_profit * max_price), 2))
        self.api.submit_order(symbol=ticker,
                         qty=amount,
                         side='buy',
                         time_in_force='gtc',
                         type='limit',
                         limit_price=max_price,
                         client_order_id='{}_buy'.format(ticker),
                         order_class='bracket',
                         stop_loss=dict(stop_price=stop_loss),
                         take_profit=dict(limit_price=take_profit)
                      )

    def sell(self, ticker, min_price, amount):
        # Selling a position you don't have is the same as shorting
        self.api.submit_order(symbol=ticker,
                              qty=amount,
                              side='sell',
                              time_in_force='gtc',
                              type='limit',
                              limit_price=min_price,
                              client_order_id='{}_sell'.format(ticker),
                              )

    def get_account_value(self):
        print(self.get_account().cash)
        return float(self.get_account().cash)

    def get_account(self):
        return self.api.get_account()

    def get_positions(self):
        self.api.list_positions()

    def get_orders(self):
        self.api.list_orders()

    def validate_order_complete(self, order_id):
        if self.api.get_order_by_client_order_id(order_id):
            return True
        else:
            return False

    def get_current_price(self, ticker):
        barset = self.api.get_barset(ticker, str(DATA_GRANULARITY), limit=1)
        bars = barset[ticker]
        return float(bars[0].c)


class AlpacaTestApi(AlpacaAPI):

    def __init__(self, lower_timestamp=None, upper_timestamp=None):
        now = datetime.now()
        self.data = {
            # Ticker: dataframe
        }
        self.trades = {
            # Ticker = [{entry, exit, amount}]
        }
        self.upper_timestamp = upper_timestamp if upper_timestamp else now
        self.lower_timestamp = lower_timestamp if lower_timestamp else (now - timedelta(days=7))

        super(AlpacaTestApi).__init__()

    def insert_ticker(self, ticker):
        self.data[ticker] = self.get_historical(
            ticker=ticker,
            upper_bound=self.upper_timestamp,
            lower_bound=self.lower_timestamp)
        data_length_set = set([len(self.data[key]) for key in self.data])
        assert len(data_length_set) == 1, "Mismatch in length of fetched historical data-sets"
        self.trades[ticker] = []

    def buy(self, ticker, max_price, amount):
        trade = {
            "entry": max_price,
            "exit": None,
            "amount": amount
        }
        self.trades[ticker].append(trade)

    def sell(self, ticker, min_price, amount):
        self.trades[ticker][-1]["exit"] = min_price
        bought_amount = self.trades[ticker][-1][amount]
        self.trades[ticker][-1][amount] = bought_amount - amount

    def get_price_at_timestamp(self, ticker, timestamp):
        data_frame = self.data[ticker]
        timed_frames = data_frame.between_time((timestamp - timedelta(seconds=DATA_GRANULARITY.value)).isoformat(), timestamp.isoformat())
        return timed_frames[0].c

    def get_current_price(self, ticker):
        raise NotImplementedError("This api-implementation is based on historical data, get current not supported")


